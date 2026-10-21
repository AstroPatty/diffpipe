import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from diffpipe.files import FileType

# Estimated peak memory of a slice processed in memory, relative to the size of its
# inputs: the input images, the output image, and column-sized temporaries.
# Measured at ~3.2x, with some headroom.
IN_MEMORY_OVERHEAD = 3.5
# Scratch space a slice's output needs relative to the size of its inputs. Outputs
# are compressed, and measured at ~0.8x on the test data, with some headroom.
SCRATCH_OVERHEAD = 1.0
# Outputs copied from scratch to their final location at once
COPY_THREADS = 4
# Every input file is stat-ed to size the slices, and each stat is a round trip on a
# network file system, so make many at once
STAT_THREADS = 32
# Environment variables an MPI launcher (srun, mpiexec) uses to hand a process its
# connection to the launcher
MPI_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_", "HYDRA_", "HYDI_")


class Budget:
    """
    A finite resource (worker slots, bytes of RAM, bytes of scratch) that slices
    must reserve before they start. A request larger than the whole budget fits
    once nothing else holds any of it, so an oversized slice can still run. Budgets
    aren't locked on their own: a Scheduler guards all of them with a single lock.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0

    def fits(self, amount: int) -> bool:
        return self.used == 0 or self.used + amount <= self.capacity


class Scheduler:
    """
    Decides which slice starts next, and reserves its share of every budget.

    Slices are started largest first, while there is room for them. When the
    largest waiting slice doesn't fit, the largest one that does starts instead,
    so memory isn't left idle while small slices wait. That could hold back a
    large slice indefinitely, so once max_skips slices have started ahead of it,
    no others start until it does.
    """

    def __init__(
        self,
        slices: list[int],
        budgets: list[tuple[Budget, Callable[[int], int]]],
        size: Callable[[int], int],
        max_skips: int,
    ):
        self.pending = sorted(slices, key=size, reverse=True)
        self.budgets = budgets
        self.max_skips = max_skips
        self.skips = 0
        self.aborted = False
        self._cond = threading.Condition()

    def _fits(self, slice) -> bool:
        return all(budget.fits(cost(slice)) for budget, cost in self.budgets)

    def _choose(self) -> int | None:
        if self._fits(self.pending[0]):
            return self.pending[0]
        if self.skips >= self.max_skips:
            return None
        return next((s for s in self.pending[1:] if self._fits(s)), None)

    def next(self) -> int | None:
        """
        Wait until a slice can start, reserve its resources and return it. Returns
        None once every slice has started, or if the run is aborted.
        """
        with self._cond:
            while True:
                if self.aborted or not self.pending:
                    return None
                slice = self._choose()
                if slice is not None:
                    break
                self._cond.wait()
            self.skips = 0 if slice == self.pending[0] else self.skips + 1
            self.pending.remove(slice)
            for budget, cost in self.budgets:
                budget.used += cost(slice)
            return slice

    def release(self, slice, budgets: list[tuple[Budget, Callable[[int], int]]]):
        with self._cond:
            for budget, cost in budgets:
                budget.used -= cost(slice)
            self._cond.notify_all()

    def abort(self):
        with self._cond:
            self.aborted = True
            self._cond.notify_all()


def detach_from_mpi_launcher():
    """
    Workers import opencosmo, which initializes MPI. Under an MPI launcher they would
    inherit a connection meant for this process alone, and abort when they try to
    use it. diffpipe doesn't use MPI, so have workers start as standalone MPI
    processes instead.
    """
    for var in [v for v in os.environ if v.startswith(MPI_LAUNCHER_ENV_PREFIXES)]:
        del os.environ[var]


def slice_input_files(step_data) -> list[Path]:
    return step_data[FileType.CORE] + step_data.get(FileType.SYNTH_CORE, [])


def output_to_scratch(slice, step_data, scratch: Path):
    """
    Point a slice's output at its own directory on scratch.
    """
    slice_dir = scratch / f"slice-{slice}"
    slice_dir.mkdir()
    staged = dict(step_data)
    staged["output_path"] = slice_dir / step_data["output_path"].name
    return staged


def stage_out(slice, step_data, staged):
    """
    Copy a finished slice's output to its final location, then remove the slice
    from scratch. The output is copied under a temporary name first, so a partial
    copy is never mistaken for a finished file.
    """
    start = time.perf_counter()
    output_path = step_data["output_path"]
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    shutil.copy(staged["output_path"], partial_path)
    partial_path.replace(output_path)
    shutil.rmtree(staged["output_path"].parent)
    logger.info(
        f"Copied output for slice {slice} to {output_path} "
        f"in {time.perf_counter() - start:.0f}s"
    )


def run_slices(
    work_orders: dict,
    process: Callable,
    n_procs: int,
    scratch: Path | None = None,
    max_memory: int | None = None,
):
    """
    Run process((slice, step_data)) for every slice in a pool of n_procs workers.

    A slice is only started once there is a free worker, memory for it (when
    max_memory is given) and scratch space for its output (when scratch is given).
    The Scheduler picks the order: largest first, with smaller slices filling any
    room the largest waiting slice can't use.
    With scratch, each slice writes its output there, and it is copied to its final
    location in the background while the worker moves on to the next slice. The
    slice's scratch space is freed once the copy is done.
    """
    slices = sorted(work_orders)
    with ThreadPoolExecutor(STAT_THREADS) as ex:
        sizes = {
            s: sum(
                ex.map(lambda p: p.stat().st_size, slice_input_files(work_orders[s]))
            )
            for s in slices
        }

    # (budget, cost of a slice). Worker slots and memory are held while a slice is
    # processed, and scratch space until its output has been copied off scratch.
    while_processing: list[tuple[Budget, Callable[[int], int]]] = [
        (Budget(n_procs), lambda s: 1)
    ]
    until_copied: list[tuple[Budget, Callable[[int], int]]] = []
    if max_memory is not None:
        while_processing.append(
            (Budget(max_memory), lambda s: int(IN_MEMORY_OVERHEAD * sizes[s]))
        )
    if scratch is not None:
        scratch_space = int(0.9 * shutil.disk_usage(scratch).free)
        until_copied.append(
            (Budget(scratch_space), lambda s: int(SCRATCH_OVERHEAD * sizes[s]))
        )
    budgets = while_processing + until_copied
    # A pool's worth of smaller slices may start ahead of the largest waiting one
    scheduler = Scheduler(slices, budgets, sizes.get, max_skips=n_procs)
    for budget, cost in budgets[1:]:
        too_big = [s for s in slices if cost(s) > budget.capacity]
        if too_big:
            logger.warning(
                f"Slices {too_big} need more memory or scratch space than is available. "
                "They will run on their own, but may still run out."
            )

    # (event, slice, staged step data, future), where event is "processed" or
    # "copied". (None, None, None, exception) if the stager fails.
    events: queue.Queue = queue.Queue()

    def stage_all(ex: ProcessPoolExecutor):
        try:
            while (s := scheduler.next()) is not None:
                step_data = work_orders[s]
                if scratch is not None:
                    step_data = output_to_scratch(s, step_data, scratch)
                future = ex.submit(process, (s, step_data))
                future.add_done_callback(
                    lambda f, s=s, d=step_data: events.put(("processed", s, d, f))
                )
        except BaseException as e:
            events.put((None, None, None, e))

    detach_from_mpi_launcher()
    # Each slice gets a fresh worker. A reused worker holds on to the memory of the
    # slices it has already processed, which the memory budget doesn't account for.
    # Leaving the copier waits for copies in flight, so if a slice fails, outputs
    # already being copied still reach their final location.
    with (
        ThreadPoolExecutor(COPY_THREADS) as copier,
        ProcessPoolExecutor(max_workers=n_procs, max_tasks_per_child=1) as ex,
    ):
        stager = threading.Thread(target=stage_all, args=(ex,), daemon=True)
        stager.start()
        try:
            remaining = len(slices)
            while remaining:
                event, s, step_data, outcome = events.get()
                if isinstance(outcome, BaseException):
                    raise outcome
                outcome.result()
                if event == "processed":
                    scheduler.release(s, while_processing)
                    if scratch is not None:
                        copy = copier.submit(stage_out, s, work_orders[s], step_data)
                        copy.add_done_callback(
                            lambda f, s=s: events.put(("copied", s, None, f))
                        )
                        continue
                else:
                    scheduler.release(s, until_copied)
                remaining -= 1
        except BaseException:
            scheduler.abort()
            ex.shutdown(cancel_futures=True)
            raise
        stager.join()

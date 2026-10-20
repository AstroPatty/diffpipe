import os
import queue
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from diffpipe.files import FileType

# Estimated peak memory of a slice processed in memory, relative to the size of its
# inputs: the input images, the output image, and column-sized temporaries.
# Measured at ~3.2x, with some headroom.
IN_MEMORY_OVERHEAD = 3.5
# Scratch space a slice needs relative to the size of its inputs: inputs plus output
SCRATCH_OVERHEAD = 2.0
# A slice is made of many small files, so copy several at once
COPY_THREADS = 8
# Environment variables an MPI launcher (srun, mpiexec) uses to hand a process its
# connection to the launcher
MPI_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_", "HYDRA_", "HYDI_")


class Budget:
    """
    A finite resource (worker slots, bytes of RAM, bytes of scratch) that slices
    must reserve before they start. A request larger than the whole budget is
    granted once nothing else holds any of it, so an oversized slice can't stall
    the run.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0
        self.aborted = False
        self._cond = threading.Condition()

    def acquire(self, amount: int) -> bool:
        with self._cond:
            self._cond.wait_for(
                lambda: self.aborted
                or self.used == 0
                or self.used + amount <= self.capacity
            )
            if self.aborted:
                return False
            self.used += amount
            return True

    def release(self, amount: int):
        with self._cond:
            self.used -= amount
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


def stage_in(slice, step_data, scratch: Path):
    """
    Copy one slice's inputs to scratch, and point its output at scratch as well.
    """
    slice_dir = scratch / f"slice-{slice}"
    slice_dir.mkdir()
    staged = dict(step_data)
    with ThreadPoolExecutor(COPY_THREADS) as ex:
        for file_type in (FileType.CORE, FileType.SYNTH_CORE):
            if file_type in step_data:
                staged[file_type] = list(
                    ex.map(
                        lambda p: Path(shutil.copy(p, slice_dir)),
                        step_data[file_type],
                    )
                )
    staged["output_path"] = slice_dir / step_data["output_path"].name
    return staged


def stage_out(step_data, staged):
    """
    Copy a finished slice's output to its final location, then remove the slice
    from scratch. The output is copied under a temporary name first, so a partial
    copy is never mistaken for a finished file.
    """
    output_path = step_data["output_path"]
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    shutil.copy(staged["output_path"], partial_path)
    partial_path.replace(output_path)
    shutil.rmtree(staged["output_path"].parent)


def run_slices(
    work_orders: dict,
    process: Callable,
    n_procs: int,
    scratch: Optional[Path] = None,
    max_memory: Optional[int] = None,
):
    """
    Run process((slice, step_data)) for every slice in a pool of n_procs workers.

    A slice is only started once there is a free worker, memory for it (when
    max_memory is given) and scratch space for it (when staging). With scratch,
    a single stager thread copies slices to scratch one at a time, each just
    before it is started, and this thread copies each output back as soon as its
    slice is finished.
    """
    slices = sorted(work_orders)
    sizes = {
        s: sum(p.stat().st_size for p in slice_input_files(work_orders[s]))
        for s in slices
    }

    # (budget, cost of a slice). When staging, allow one slice beyond the number of
    # workers, so the next slice is already on scratch when a worker frees up.
    budgets: list[tuple[Budget, Callable[[int], int]]] = [
        (Budget(n_procs + (scratch is not None)), lambda s: 1)
    ]
    if max_memory is not None:
        budgets.append(
            (Budget(max_memory), lambda s: int(IN_MEMORY_OVERHEAD * sizes[s]))
        )
    if scratch is not None:
        scratch_space = int(0.9 * shutil.disk_usage(scratch).free)
        budgets.append(
            (Budget(scratch_space), lambda s: int(SCRATCH_OVERHEAD * sizes[s]))
        )
    for budget, cost in budgets[1:]:
        too_big = [s for s in slices if cost(s) > budget.capacity]
        if too_big:
            logger.warning(
                f"Slices {too_big} need more memory or scratch space than is available. "
                "They will run on their own, but may still run out."
            )

    finished: queue.Queue = queue.Queue()

    def stage_all(ex: ProcessPoolExecutor):
        try:
            for s in slices:
                for budget, cost in budgets:
                    if not budget.acquire(cost(s)):
                        return
                step_data = work_orders[s]
                if scratch is not None:
                    start = time.perf_counter()
                    step_data = stage_in(s, step_data, scratch)
                    logger.info(
                        f"Copied slice {s} to scratch "
                        f"({sizes[s] / 1e9:.1f} GB in {time.perf_counter() - start:.0f}s)"
                    )
                future = ex.submit(process, (s, step_data))
                future.add_done_callback(
                    lambda f, s=s, d=step_data: finished.put((s, d, f))
                )
        except BaseException as e:
            finished.put((None, None, e))

    detach_from_mpi_launcher()
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        stager = threading.Thread(target=stage_all, args=(ex,), daemon=True)
        stager.start()
        try:
            for _ in slices:
                s, step_data, outcome = finished.get()
                if isinstance(outcome, BaseException):
                    raise outcome
                outcome.result()
                if scratch is not None:
                    start = time.perf_counter()
                    stage_out(work_orders[s], step_data)
                    logger.info(
                        f"Copied output for slice {s} to {work_orders[s]['output_path']} "
                        f"in {time.perf_counter() - start:.0f}s"
                    )
                for budget, cost in budgets:
                    budget.release(cost(s))
        except BaseException:
            for budget, _ in budgets:
                budget.abort()
            ex.shutdown(cancel_futures=True)
            raise
        stager.join()

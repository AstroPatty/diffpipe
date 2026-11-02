import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from diffpipe.files import FileType

# Estimated peak memory of a slice processed in memory, relative to the size of its
# inputs: the input images, the output image, and column-sized temporaries.
# Measured at ~3.2x, with some headroom.
IN_MEMORY_OVERHEAD = 3.5
# Smaller slices, per worker, that may start ahead of the largest waiting slice
# before the scheduler stops starting others until it fits
BACKFILL_PER_WORKER = 4
# Every input file is stat-ed to size the slices, and each stat is a round trip on a
# network file system, so make many at once
STAT_THREADS = 32
# Environment variables an MPI launcher (srun, mpiexec) uses to hand a process its
# connection to the launcher
MPI_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_", "HYDRA_", "HYDI_")


class Budget:
    """
    A finite resource (worker slots, bytes of RAM) that slices must reserve before
    they start. A request larger than the whole budget fits once nothing else holds
    any of it, so an oversized slice can still run. Budgets aren't locked on their
    own: a Scheduler guards all of them with a single lock.
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


def run_slices(
    work_orders: dict,
    process: Callable,
    n_procs: int,
    max_memory: int | None = None,
):
    """
    Run process((slice, step_data)) for every slice in a pool of n_procs workers.

    A slice is only started once there is a free worker and, when max_memory is
    given, memory for it. The Scheduler picks the order: largest first, with smaller
    slices filling any room the largest waiting slice can't use.
    """
    slices = sorted(work_orders)
    with ThreadPoolExecutor(STAT_THREADS) as ex:
        sizes = {
            s: sum(
                ex.map(lambda p: p.stat().st_size, slice_input_files(work_orders[s]))
            )
            for s in slices
        }

    # (budget, cost of a slice), held while the slice is processed
    budgets: list[tuple[Budget, Callable[[int], int]]] = [
        (Budget(n_procs), lambda s: 1)
    ]
    if max_memory is not None:
        budgets.append(
            (Budget(max_memory), lambda s: int(IN_MEMORY_OVERHEAD * sizes[s]))
        )
    # A few pools' worth of smaller slices may start ahead of the largest waiting one.
    # Fewer drains the pool when the waiting slice would soon fit anyway, more lets
    # the largest slice be held back for longer.
    scheduler = Scheduler(
        slices, budgets, sizes.get, max_skips=BACKFILL_PER_WORKER * n_procs
    )
    for budget, cost in budgets[1:]:
        too_big = [s for s in slices if cost(s) > budget.capacity]
        if too_big:
            logger.warning(
                f"Slices {too_big} need more memory than is available. "
                "They will run on their own, but may still run out."
            )

    # (slice, future) as each slice finishes, or (None, exception) if the submitter
    # fails
    finished: queue.Queue = queue.Queue()

    def submit_all(ex: ProcessPoolExecutor):
        try:
            while (s := scheduler.next()) is not None:
                future = ex.submit(process, (s, work_orders[s]))
                future.add_done_callback(lambda f, s=s: finished.put((s, f)))
        except BaseException as e:
            finished.put((None, e))

    detach_from_mpi_launcher()
    # Each slice gets a fresh worker. A reused worker holds on to the memory of the
    # slices it has already processed, which the memory budget doesn't account for.
    with ProcessPoolExecutor(max_workers=n_procs, max_tasks_per_child=1) as ex:
        submitter = threading.Thread(target=submit_all, args=(ex,), daemon=True)
        submitter.start()
        try:
            for _ in slices:
                s, outcome = finished.get()
                if isinstance(outcome, BaseException):
                    raise outcome
                outcome.result()
                scheduler.release(s, budgets)
        except BaseException:
            scheduler.abort()
            ex.shutdown(cancel_futures=True)
            raise
        submitter.join()

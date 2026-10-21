import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from diffpipe import BadInputError
from diffpipe.data import process_slice
from diffpipe.files import build_work_orders
from diffpipe.header import get_simulation_header_data
from diffpipe.pipeline import run_slices
from diffpipe.schema import SCHEMA_FILE_NAME

DATA_FOLDER = click.Path(
    exists=True, file_okay=False, readable=True, resolve_path=True, path_type=Path
)

OUTPUT_FOLDER = click.Path(
    file_okay=False, writable=True, resolve_path=True, path_type=Path
)


@click.group()
def diffpipe():
    pass


@click.command()
@click.argument("input_folder", required=True, type=DATA_FOLDER)
@click.argument("output_folder", required=True, type=OUTPUT_FOLDER)
@click.option(
    "--scratch",
    required=False,
    type=OUTPUT_FOLDER,
    help="Fast local storage to work in. Slices are copied here one at a time as they "
    "are needed, and each output is copied back as soon as it is finished",
)
@click.option("--pre-copy", "-p", is_flag=True, hidden=True)
@click.option(
    "--in-memory/--no-in-memory",
    default=True,
    help="Read each input file and build each output file in memory, so the "
    "filesystem only sees large sequential reads and writes. Defaults to on",
)
@click.option(
    "--max-memory",
    required=False,
    type=float,
    help="Memory (in GB) that slices being processed in memory may use at once. "
    "Defaults to 75% of physical memory",
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Overwite any existing files in the output directory",
)
@click.option(
    "--n-procs",
    "-n",
    is_flag=False,
    required=False,
    type=int,
    help="Number of processes. If not provided, inferred from execution environment",
)
@click.option(
    "--index-depth",
    "-d",
    is_flag=False,
    required=False,
    default=10,
    type=int,
    help="Depth of spatial index. Will use a healpix decomposition with nside = 2**index_depth. Defaults to 10",
)
@click.option(
    "--simulation",
    "-s",
    is_flag=False,
    required=False,
    default="LastJourney",
    type=str,
    help="Underlying simulation this catalog was built on. Defaults to LastJourney",
)
def run(
    input_folder: Path,
    output_folder: Path,
    scratch: Optional[Path],
    pre_copy: bool,
    in_memory: bool,
    max_memory: Optional[float],
    overwrite: bool,
    n_procs: Optional[int],
    index_depth: int,
    simulation: str,
):
    """
    Convert diffsky catalogs in CORE_FOLDER into opencosmo-formatted files,
    placing them in OUTPUT_FOLDER. Can optionally also convert associated
    synthetic-core-based catalogs.
    """

    try:
        _ = get_simulation_header_data(simulation)
    except BadInputError:
        logger.critical("Terminating due to previous error")
        sys.exit(1)

    if pre_copy:
        logger.warning(
            "--pre-copy is deprecated and has no effect. With --scratch, slices are "
            "copied to scratch one at a time as they are needed"
        )

    scratch_path = scratch
    if scratch is not None:
        scratch_path = scratch / "diffpipe"
        scratch_path.mkdir(parents=False, exist_ok=False)

    work_orders = build_work_orders(input_folder, output_folder, simulation, overwrite)
    logger.info(f"Found data for redshift slices {list(work_orders.keys())}")
    if n_procs is None:
        n_procs = multiprocessing.cpu_count()

    memory_budget = None
    if in_memory:
        if max_memory is None:
            physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            memory_budget = int(0.75 * physical_memory)
        else:
            memory_budget = int(max_memory * 1e9)
        logger.info(
            f"Processing slices in memory, using up to {memory_budget / 1e9:.0f} GB"
        )

    logger.info(f"Running conversion with {n_procs} processes")
    run_step_f = partial(
        run_step, index_depth=index_depth, simulation=simulation, in_memory=in_memory
    )
    schema_path = output_folder / SCHEMA_FILE_NAME
    try:
        run_slices(work_orders, run_step_f, n_procs, scratch_path, memory_budget)
    except BadInputError:
        logger.critical("Terminating due to previous error")
        sys.exit(1)
    finally:
        schema_path.unlink(missing_ok=True)
    if scratch_path is not None:
        scratch_path.rmdir()

    logger.success("All files processed!")


def run_step(step_data, index_depth: int, simulation: str, in_memory: bool):
    step, data = step_data
    return process_slice(step, data, index_depth, simulation, in_memory)


diffpipe.add_command(run)

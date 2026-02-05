import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from diffpipe import BadInputError
from diffpipe.data import process_slice
from diffpipe.files import build_work_orders
from diffpipe.header import get_simulation_header_data

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
    core_folder: Path,
    output_folder: Path,
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

    work_orders = build_work_orders(core_folder, output_folder, simulation, overwrite)
    logger.info(f"Found data for redshift slices {list(work_orders.keys())}")
    if n_procs is None:
        n_procs = multiprocessing.cpu_count()
    logger.info(f"Running conversion with {n_procs} processes")
    run_step_f = partial(run_step, index_depth=index_depth, simulation=simulation)
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        futures = [ex.submit(run_step_f, item) for item in work_orders.items()]
        for fut in as_completed(futures):
            fut.result()

    logger.success("All files processed!")


def run_step(step_data, index_depth: int, simulation: str):
    step, data = step_data
    return process_slice(step, data, index_depth, simulation)


diffpipe.add_command(run)

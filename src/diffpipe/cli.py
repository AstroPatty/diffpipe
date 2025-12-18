import multiprocessing
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from diffpipe.data import process_slice
from diffpipe.files import build_work_orders

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
@click.argument("core_folder", required=True, type=DATA_FOLDER)
@click.option(
    "--synth-core-folder",
    "-s",
    is_flag=False,
    required=False,
    type=DATA_FOLDER,
    help="Path to folder containing synthetic core catalogs",
)
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
def run(
    core_folder: Path,
    synth_core_folder: Optional[Path],
    output_folder: Path,
    overwrite: bool,
    n_procs: Optional[int],
):
    """
    Convert diffsky catalogs in CORE_FOLDER into opencosmo-formatted files,
    placing them in OUTPUT_FOLDER. Can optionally also convert associated
    synthetic-core-based catalogs.
    """

    work_orders = build_work_orders(
        core_folder, synth_core_folder, output_folder, overwrite
    )
    logger.info(f"Found data for redshift slices {list(work_orders.keys())}")
    if n_procs is None:
        n_procs = multiprocessing.cpu_count()
    logger.info(f"Running conversion with {n_procs} processes")
    with multiprocessing.Pool(n_procs) as pool:
        pool.map(run_step, work_orders.items())

    logger.success("All files processed!")


def run_step(step_data):
    step, data = step_data
    return process_slice(step, data)


diffpipe.add_command(run)

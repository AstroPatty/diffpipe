from pathlib import Path
from typing import Optional

import click

from diffpipe.files import build_file_lists

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
@click.option("--synth-core-folder", is_flag=False, required=False, type=DATA_FOLDER)
@click.argument("output_folder", required=True, type=OUTPUT_FOLDER)
def run(core_folder: Path, synth_core_folder: Optional[Path], output_folder: Path):
    file_lists = build_file_lists(core_folder, synth_core_folder)
    print(file_lists)


diffpipe.add_command(run)

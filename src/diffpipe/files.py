import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Iterable

from loguru import logger

from diffpipe.header import get_simulation_header_data
from diffpipe.schema import SCHEMA_FILE_NAME

# lc_cores-{slice}.{pixel}.diffsky_gals[.synthetic_halos].hdf5
CATALOG_FILE_PATTERN = re.compile(
    r"lc_cores-(\d+)\.(\d+)\.diffsky_gals(\.synthetic_halos)?\.hdf5"
)


class FileType(Enum):
    CORE = "cores"
    SYNTH_CORE = "synth_cores"


def build_work_orders(
    input_folder: Path,
    output_folder: Path,
    simulation: str,
    overwrite: bool,
    restart: bool = False,
):
    """
    Build the work order for every slice to convert. With restart, slices whose
    output already exists are left out, since outputs only get their final name
    once they are complete.
    """
    _ = get_simulation_header_data(simulation)
    files_by_slice = build_file_lists(input_folder)

    # The contents of the files are verified by each slice as it is converted,
    # against a schema written by the first one. Remove any left by an earlier run.
    schema_path = output_folder / SCHEMA_FILE_NAME
    schema_path.unlink(missing_ok=True)

    # Every slice is listed in each output's header, including any already done
    all_slices = set(files_by_slice.keys())
    output_paths, completed = get_output_paths(
        output_folder, all_slices, overwrite or restart
    )
    for slice in all_slices:
        files_by_slice[slice]["output_path"] = output_paths[slice]
        files_by_slice[slice]["all_slices"] = all_slices
        files_by_slice[slice]["schema_path"] = schema_path

    if restart and completed:
        logger.info(f"Skipping slices {sorted(completed)}, which are already done")
        for slice in completed:
            del files_by_slice[slice]
    return files_by_slice


def get_output_paths(output_folder: Path, slices: Iterable[int], allow_existing: bool):
    """
    Get the output path of every slice, and the slices whose output already exists.
    Exits if any do, unless allow_existing.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    # List the folder once, rather than checking for each output separately
    existing = {p.name for p in output_folder.glob("*.hdf5")}
    output_paths = {
        slice: output_folder / f"lc_cores-{slice}.diffsky_gals.hdf5" for slice in slices
    }
    completed = {slice for slice, p in output_paths.items() if p.name in existing}
    if completed and not allow_existing:
        logger.critical(
            f"Found existing catalog files for slices {sorted(completed)} in "
            f"{output_folder}. Run with --overwrite to replace them, or --restart to "
            "keep them and convert the remaining slices"
        )
        sys.exit()
    if len(existing) > len(completed):
        logger.warning(
            "Output folder has existing hdf5 files, but there are no name clashes. Continuing anyway..."
        )

    return output_paths, completed


def build_file_lists(input_folder):
    all_files_by_slice = get_files_by_slice(input_folder)
    core_files_by_slice, synth_core_files_by_slice = split_slice_files(
        all_files_by_slice
    )

    verify_file_lists(core_files_by_slice, synth_core_files_by_slice)
    output = defaultdict(dict)
    for slice, core_files in core_files_by_slice.items():
        output[slice][FileType.CORE] = core_files
        if synth_core_files_by_slice:
            output[slice][FileType.SYNTH_CORE] = synth_core_files_by_slice[slice]
    return output


def verify_file_lists(
    core_files_by_slice: dict[int, list[Path]],
    synthetic_core_files_by_slice: dict[int, list[Path]],
):
    if not core_files_by_slice or all(
        len(paths) == 0 for paths in core_files_by_slice.values()
    ):
        logger.critical("Found no core files to convert!")
        sys.exit()

    if not synthetic_core_files_by_slice:
        return True

    if set(core_files_by_slice.keys()) != set(synthetic_core_files_by_slice.keys()):
        logger.critical(
            "The simulated cores and synthetic cores do not have the same set of redshift slices!"
        )
        sys.exit()

    for slice, core_files in core_files_by_slice.items():
        synthetic_core_files = synthetic_core_files_by_slice[slice]
        core_pixels = set(map(get_file_pixel, core_files))
        synthetic_core_pixels = set(map(get_file_pixel, synthetic_core_files))

        if core_pixels != synthetic_core_pixels:
            logger.critical(
                f"Core and synthetic core files for slice {slice} do not cover the same pixels!"
            )
            sys.exit()
    return True


def get_files_by_slice(folder: Path) -> dict[int, list[Path]]:
    # Files are found by name alone, since opening each one is slow on a network
    # file system. Whether they really are catalog files is checked on conversion.
    files_by_slice = defaultdict(list)
    for file in Path(folder).glob("*.hdf5"):
        match = CATALOG_FILE_PATTERN.fullmatch(file.name)
        if match is not None:
            files_by_slice[int(match.group(1))].append(file)
    return dict(files_by_slice)


def split_slice_files(files_by_slice: dict[int, list[Path]]):
    core_output = {}
    synth_core_output = {}
    for slice, files in files_by_slice.items():
        synth_core_files = list(
            filter(lambda p: p.stem.endswith("diffsky_gals.synthetic_halos"), files)
        )
        core_files = list(filter(lambda p: p.stem.endswith("diffsky_gals"), files))
        core_output[slice] = core_files
        if synth_core_files:
            synth_core_output[slice] = synth_core_files
    return core_output, synth_core_output


def get_file_pixel(path: Path):
    return int(path.stem.split(".")[1])

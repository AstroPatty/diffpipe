import sys
from collections import defaultdict
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

import h5py
from loguru import logger

from diffpipe.header import get_simulation_header_data, verify_incoming_metadata


class FileType(Enum):
    CORE = 1
    SYNTH_CORE = 2


def build_work_orders(
    core_folder: Path,
    synthetic_core_folders: Optional[Path],
    output_folder: Path,
    simulation: str,
    overwrite: bool,
):
    _ = get_simulation_header_data(simulation)

    files_by_slice = build_file_lists(core_folder, synthetic_core_folders)
    all_slices = set(files_by_slice.keys())
    output_paths = get_output_paths(output_folder, all_slices, overwrite)
    for slice in all_slices:
        files_by_slice[slice]["output_path"] = output_paths[slice]
        files_by_slice[slice]["all_slices"] = all_slices

    return files_by_slice


def get_output_paths(output_folder: Path, slices: Iterable[int], overwrite: bool):
    output_folder.mkdir(parents=True, exist_ok=True)
    existing_hdf5_files = list(output_folder.glob("*.hdf5"))
    if existing_hdf5_files:
        logger.warning(
            "Output folder has existing hdf5 files, but there are no name clashes. Continuing anyway..."
        )

    output_paths = {}
    for slice in slices:
        output_file_path = output_folder / Path(f"lc_cores-{slice}.diffsky_gals.hdf5")
        if output_file_path.exists() and not overwrite:
            logger.critical(
                f"Found an existing catalog file at {output_file_path}. Run with --overwrite to ignore"
            )
            sys.exit()
        output_paths[slice] = output_file_path

    return output_paths


def build_file_lists(core_folder: Path, synthetic_core_folder: Optional[Path]):
    synthetic_core_files_by_slice = {}
    core_files_by_slice = get_files_by_slice(core_folder)
    if synthetic_core_folder is not None:
        synthetic_core_files_by_slice = get_files_by_slice(synthetic_core_folder)

    verify_file_lists(core_files_by_slice, synthetic_core_files_by_slice)
    output = defaultdict(dict)
    for slice, core_files in core_files_by_slice.items():
        output[slice][FileType.CORE] = core_files
        if synthetic_core_files_by_slice:
            output[slice][FileType.SYNTH_CORE] = synthetic_core_files_by_slice[slice]
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

    all_core_files = list(chain(*[files for files in core_files_by_slice.values()]))
    all_synth_core_files = list(
        chain(*[files for files in synthetic_core_files_by_slice.values()])
    )
    verify_incoming_metadata(all_core_files + all_synth_core_files)
    logger.success("Metadata is consistent across files!")

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
    core_files = filter(is_catalog_file, Path(folder).glob("*.hdf5"))
    core_files = list(core_files)

    redshift_slices = set()
    for file in core_files:
        slice = int(
            file.stem.split("-")[1].split(".")[0]
        )  # we should include this information in the metadata
        redshift_slices.add(slice)

    files_by_slice = {}

    for slice in redshift_slices:
        core_slice_files = list(
            filter(lambda f: f"lc_cores-{slice}" in f.stem, core_files)
        )
        files_by_slice[slice] = core_slice_files
    return files_by_slice


def get_file_pixel(path: Path):
    return int(path.stem.split(".")[1])


def is_catalog_file(path: Path) -> bool:
    with h5py.File(path) as f:
        if "data" not in f.keys() or "metadata" not in f.keys():
            return False
        elif "mock_version_name" not in f["metadata"].attrs.keys():
            return False
        return True

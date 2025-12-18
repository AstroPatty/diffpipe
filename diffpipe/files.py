from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

import h5py


class FileType(Enum):
    CORE = 1
    SYNTH_CORE = 2


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
        raise ValueError("Found no core files to convert!")
    if not synthetic_core_files_by_slice:
        return True

    if set(core_files_by_slice.keys()) != set(synthetic_core_files_by_slice.keys()):
        raise ValueError(
            "The simulated cores and synthetic cores do not have the same set of redshift slices!"
        )

    for slice, core_files in core_files_by_slice.items():
        synthetic_core_files = synthetic_core_files_by_slice[slice]
        core_pixels = set(map(get_file_pixel, core_files))
        synthetic_core_pixels = set(map(get_file_pixel, synthetic_core_files))

        if core_pixels != synthetic_core_pixels:
            raise ValueError(
                f"Core and synthetic core files for slice {slice} do not cover the same pixels!"
            )
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


def get_file_pixel(path: Path):
    return int(path.stem.split(".")[1])


def is_catalog_file(path: Path) -> bool:
    with h5py.File(path) as f:
        if "data" not in f.keys() or "metadata" not in f.keys():
            return False
        elif "mock_version_name" not in f["metadata"].attrs.keys():
            return False
        return True

import os
from multiprocessing import Pool
from pathlib import Path

from diffpipe.data import write_files
from diffpipe.header import copy_header

MAX_LEVEL = 7

output_dir = Path("/Volumes/workspace/data/LastJourney/synthetic_galaxies/toolkit")
HEADER_SOURCE = "header.hdf5"


def write(data):
    slice, config = data
    output = config["output"]
    all_slices = config.pop("all_slices")
    write_files(slice=slice, files=config, max_level=MAX_LEVEL)
    copy_header(HEADER_SOURCE, output, slice, all_slices)


def process_all_files(core_folder, synth_core_folder):
    if not output_dir.exists():
        os.mkdir(output_dir)
    core_files = list(Path(core_folder).glob("*.hdf5"))
    synth_core_files = list(Path(synth_core_folder).glob("*.hdf5"))
    core_slices = set()
    synth_core_slices = set()
    for file in core_files:
        slice = int(file.stem.split("-")[1].split(".")[0])
        core_slices.add(slice)

    for file in synth_core_files:
        slice = int(file.stem.split("-")[1].split(".")[0])
        synth_core_slices.add(slice)

    if synth_core_slices and not core_slices == synth_core_slices:
        raise ValueError(
            "Synthetic cores and halo cores do not have all the same slices!"
        )

    files_by_slice = {}
    for slice in core_slices:
        core_slice_files = list(filter(lambda f: str(slice) in f.stem, core_files))
        synth_core_slice_files = list(
            filter(lambda f: str(slice) in f.stem, synth_core_files)
        )
        files_by_slice[slice] = {
            "cores": core_slice_files,
            # "synthetic_cores": synth_core_slice_files,
            "output": output_dir / Path(f"lc_cores-{slice}.diffsky_gals.hdf5"),
            "all_slices": core_slices,
        }

    with Pool(8) as p:
        p.map(write, files_by_slice.items())


if __name__ == "__main__":
    process_all_files(
        "/Volumes/workspace/data/LastJourney/synthetic_galaxies/original",
        "data/synthetic_cores",
    )

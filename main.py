import os
from multiprocessing import Pool
from pathlib import Path

from diffpipe.data import write_files
from diffpipe.header import copy_header

MAX_LEVEL = 10

output_dir = Path("/lcrc/project/halotools/prwells/diffsky/11072025")
HEADER_SOURCE = "header.hdf5"


def write(data):
    slice, config = data
    output = config["output"]
    all_slices = config.pop("all_slices")
    core_files = config.pop("cores")
    core_files.sort()

    pixels_with_data = write_files(
        slice=slice, core_files=core_files, output_path=output, max_level=MAX_LEVEL
    )
    copy_header(
        HEADER_SOURCE, output, slice, all_slices, pixels_with_data, 2**MAX_LEVEL
    )


def process_all_files(core_folder, synthetic_core_folder):
    if not output_dir.exists():
        os.mkdir(output_dir)
    core_files = list(Path(core_folder).glob("lc_cores*.hdf5"))
    core_slices = set()
    for file in core_files:
        slice = int(file.stem.split("-")[1].split(".")[0])
        core_slices.add(slice)

    files_by_slice = {}
    for slice in core_slices:
        core_slice_files = list(filter(lambda f: str(slice) in f.stem, core_files))
        files_by_slice[slice] = {
            "cores": core_slice_files,
            "output": output_dir / Path(f"lc_cores-{slice}.diffsky_gals.hdf5"),
            "all_slices": core_slices,
        }

    with Pool(len(files_by_slice)) as p:
        p.map(write, files_by_slice.items())


if __name__ == "__main__":
    process_all_files("/lcrc/project/halotools/random_data/1107/smdpl_dr1")

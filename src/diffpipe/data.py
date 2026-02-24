import shutil
import sys
from pathlib import Path
from typing import Optional

import astropy.units as u
import h5py
import hdf5plugin
import healpy as hp
import numpy as np
from astropy.cosmology import units as cu
from loguru import logger

from diffpipe.files import FileType
from diffpipe.header import write_opencosmo_header
from diffpipe.index import (
    combine_counts,
    get_counts,
    make_combined_file_map,
    verify_index,
)

COMPRESSION = hdf5plugin.Blosc2(
    cname="lz4", clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE
)

LITTLE_H = 0.6766
u.add_enabled_units(cu)

COLUMNS_TO_SKIP = ["ra", "dec"]
COLUMN_RENAMES = {"ra_nfw": "ra", "dec_nfw": "dec"}
UNIT_MAP = {"ra_obs": u.deg, "dec_obs": u.deg}


def process_slice(slice, step_data, index_depth, simulation):
    logger.info(f"Working on slice {slice}")
    core_files = step_data[FileType.CORE]  # required
    output_path = step_data["output_path"]  # required
    all_slices = step_data["all_slices"]  # required
    global_offset = step_data["global_offset"]
    scratch_path: Optional[Path] = step_data.get("scratch_path")
    synth_core_files = step_data.get(FileType.SYNTH_CORE)  # optional

    if scratch_path is not None:
        is_in_scratch = [f.is_relative_to(scratch_path) for f in core_files]
        if synth_core_files is not None:
            is_in_scratch.extend(
                f.is_relative_to(scratch_path) for f in synth_core_files
            )

        scratch_output = scratch_path / "output"
        scratch_output.mkdir(parents=False, exist_ok=True)
        file_output_path = scratch_output / output_path.name

        if not any(is_in_scratch):
            logger.info(f"Copying data for slice {slice} to scratch at {scratch_path}")
            for file in core_files:
                shutil.copy(file, scratch_path)
            core_files = [scratch_path / f.name for f in core_files]
            if synth_core_files is not None:
                for file in synth_core_files:
                    shutil.copy(file, scratch_path)
                synth_core_files = [scratch_path / f.name for f in synth_core_files]
        elif not all(is_in_scratch):
            raise ValueError(
                "Found a mixture of slices both in and out of scratch... Aborting!"
            )

    else:
        file_output_path = output_path

    pixels_with_data = write_files(
        slice,
        core_files,
        synth_core_files,
        file_output_path,
        index_depth,
        global_offset,
    )

    write_opencosmo_header(
        core_files[0],
        file_output_path,
        simulation,
        slice,
        all_slices,
        pixels_with_data,
        index_depth,
    )
    if scratch_path is not None:
        shutil.copy(file_output_path, output_path)
        file_output_path.unlink()
        all_files = core_files + synth_core_files or []
        for file in all_files:
            assert file.is_relative_to(scratch_path)
            file.unlink()

    logger.success(f"Successfully wrote data for slice {slice}")


def get_columns_in_group(group: h5py.Group, verify_length=True):
    columns = {}

    for name, child in group.items():
        if isinstance(child, h5py.Dataset):
            columns[name] = child
        elif "unlensed" not in name:
            raise ValueError("Can only handle unlensed groups as children of data")
        else:
            for name_, grandchild in child.items():
                columns[f"{name_}_unlensed"] = grandchild
    lengths = set(len(col) for col in columns.values())
    if not len(lengths) == 1:
        raise ValueError("All columns in a single group must be the same length!")
    return columns


def verify_column_consistency(files: list[Path]):
    """
    Verify that all files have the same columns, and for each column, that:
    1. The shapes are compatible (identical after 0th axis)
    2. The data types are compatible
    """

    with h5py.File(files[0]) as reference:
        columns = get_columns_in_group(reference["data"])

        reference_shapes = {colname: col.shape[1:] for colname, col in columns.items()}
        reference_dtypes = {colname: col.dtype for colname, col in columns.items()}
        reference_attrs = {colname: dict(col.attrs) for colname, col in columns.items()}

    for file_path in files[1:]:
        with h5py.File(file_path) as to_compare:
            columns = get_columns_in_group(to_compare["data"])
            shapes = {colname: col.shape[1:] for colname, col in columns.items()}
            dtypes = {colname: col.dtype for colname, col in columns.items()}
            attrs = {colname: dict(col.attrs) for colname, col in columns.items()}
            if set(shapes.keys()) != set(reference_shapes.keys()):
                logger.critical("Files do not all have the same columns!")
                sys.exit(1)
            if shapes != reference_shapes:
                logger.critical("Column shapes are not consistent across files!")
                sys.exit(1)
            if attrs != reference_attrs:
                logger.critical("Column attributes are not consistent across files!")
                sys.exit(1)

            for name, rdtype in reference_dtypes.items():
                target_dtype = dtypes[name]
                promoted_type = np.promote_types(rdtype, target_dtype)
                if promoted_type not in [rdtype, target_dtype]:
                    # Columns may have different precision (numpy will automatically promote later)
                    # But we are avoiding situations where we have eg. int and float
                    logger.critical("Column dtypes are not consistent across files!")
                    sys.exit(1)

    logger.success("Columns are consistent across files!")


def write_files(
    slice, core_files, synth_core_files, output_path, max_level, global_offset
):
    counts = {}

    counts[FileType.CORE] = {f: get_counts(max_level, f) for f in core_files}
    file_map = {}
    file_map[FileType.CORE] = make_combined_file_map(max_level, core_files)
    if synth_core_files is not None:
        counts[FileType.SYNTH_CORE] = {
            f: get_counts(max_level, f) for f in synth_core_files
        }
        file_map[FileType.SYNTH_CORE] = make_combined_file_map(
            max_level, synth_core_files
        )

    files = {FileType.CORE: core_files}
    if synth_core_files is not None:
        files[FileType.SYNTH_CORE] = synth_core_files

    allocate_file(output_path, files, file_map)
    with h5py.File(output_path, "a") as target:
        write_single_file(max_level, files, target, file_map, global_offset)
        pixels_with_data = write_indices(target, counts, max_level)
        if FileType.SYNTH_CORE in files and len(files) > 1:
            if_group = target.require_group(f"{FileType.SYNTH_CORE.value}/load/if")
            if_group.attrs["synth_cores"] = True

    target.close()
    with h5py.File(output_path) as target:
        if len(files) > 1:
            for group_name, group in target.items():
                if group_name == "header":
                    continue
                verify_index(group, max_level)
        else:
            verify_index(target, max_level)
    return pixels_with_data


def write_indices(target, counts, max_level):
    if len(counts) == 1:
        file_type = next(iter(counts.keys()))
        return write_index(target, counts[file_type], max_level)

    pixels_with_data = np.array([], dtype=int)
    for file_type, file_type_counts in counts.items():
        output_group = target[file_type.value]
        file_type_pixels = write_index(output_group, file_type_counts, max_level)
        pixels_with_data = np.union1d(pixels_with_data, file_type_pixels)
    return pixels_with_data


def write_index(output_group, counts, max_level):
    index_group = output_group.require_group("index")
    nside = 2**max_level
    npix = hp.nside2npix(nside)
    starts = np.zeros(npix, dtype=np.int32)
    sizes = np.zeros(npix, dtype=np.int32)
    combined_counts = combine_counts(counts)
    pixels = np.sort(np.fromiter(combined_counts.keys(), dtype=int))
    min_ = pixels[0]
    max_ = pixels[-1]

    running_total = 0

    for pixel in range(min_, max_ + 1):
        pixel_count = combined_counts.get(pixel, 0)
        sizes[pixel] = pixel_count
        starts[pixel] = running_total
        running_total += pixel_count
    starts[max_ + 1 :] = running_total
    pixels_with_data = np.where(sizes > 0)[0]

    for level in range(max_level, -1, -1):
        level_group = index_group.require_group(f"level_{level}")
        level_group.create_dataset("start", data=starts, compression=COMPRESSION)
        level_group.create_dataset("size", data=sizes, compression=COMPRESSION)
        sizes = sizes.reshape(-1, 4).sum(axis=1)
        starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    return pixels_with_data


def write_single_file(max_level, source_files, file_target, file_map, global_offset):
    if len(source_files) == 1:
        file_type = next(iter(source_files.keys()))
        sources = [h5py.File(s) for s in source_files[file_type]]
        known_columns = [
            get_columns_in_group(sf["data"], verify_length=False) for sf in source_files
        ]
        for column_name in known_columns[0].keys():
            column_sources = [kc[column_name] for kc in known_columns]
            write_column(
                file_target["data"], column_sources, column_name, file_map[file_type]
            )
        file_target["data"]["gal_id"][:] = global_offset + np.arange(
            0, len(file_target["data"]["gal_id"])
        )

        return

    local_offset = 0
    for file_type, file_type_sources in source_files.items():
        sources = [h5py.File(s) for s in source_files[file_type]]
        known_columns = [
            get_columns_in_group(sf["data"], verify_length=False) for sf in sources
        ]
        dataset_group = file_target[f"{file_type.value}/data"]

        for column_name in known_columns[0].keys():
            column_sources = [kc[column_name] for kc in known_columns]
            write_column(
                dataset_group,
                column_sources,
                column_name,
                file_map[file_type],
            )
        dataset_length = len(dataset_group["gal_id"])
        ids = global_offset + local_offset + np.arange(dataset_length)
        print(ids)

        dataset_group["gal_id"][:] = ids
        print(dataset_group["gal_id"])

        local_offset += dataset_length


def write_column(output_data_group, sources, column_name, full_map):
    data = np.concat([source[:] for source in sources])
    attributes = dict(sources[0].attrs)
    if column_name in COLUMNS_TO_SKIP:
        return
    if column_name in UNIT_MAP:
        attributes["unit"] = str(UNIT_MAP[column_name])

    output_column_name = COLUMN_RENAMES.get(column_name, column_name)
    if attributes.get("unit", "") != "":
        unit = u.Unit(attributes["unit"])

        try:
            bases = unit.bases
            index = bases.index(cu.littleh)
            h_power = unit.powers[index]
        except (ValueError, AttributeError):
            h_power = 0

        if h_power != 0:
            data = data / LITTLE_H**h_power
            unit = unit / cu.littleh**h_power
            attributes["unit"] = str(unit)

    output_ds = output_data_group[output_column_name]
    output_ds[:] = data[full_map]
    output_ds.attrs.update(attributes)
    output_data_group.file.flush()


def allocate_dataset_group(group, source_files, total_length):
    data_group = group.require_group("data")
    column_metadata_source = source_files[0]
    shapes = {}
    dtypes = {}
    with h5py.File(column_metadata_source) as source_f:
        columns = get_columns_in_group(source_f["data"], verify_length=False)
        for colname, col in columns.items():
            shapes[colname] = col.shape[1:]
            dtypes[colname] = col.dtype
    dtypes["gal_id"] = np.int64
    shapes["gal_id"] = ()

    for col, dtype in dtypes.items():
        if col in COLUMNS_TO_SKIP:
            continue
        name = COLUMN_RENAMES.get(col, col)
        shape = (total_length,) + shapes[col]

        data_group.create_dataset(
            name, shape=shape, dtype=dtype, compression=COMPRESSION
        )


def allocate_file(output_path, source_files, file_map):
    with h5py.File(output_path, "w") as f_output:
        if len(file_map) == 1:
            file_type = next(iter(file_map.keys()))
            allocate_dataset_group(
                f_output, source_files[file_type], len(file_map[file_type])
            )
        else:
            for file_type, file_type_map in file_map.items():
                group = f_output.require_group(file_type.value)
                allocate_dataset_group(
                    group, source_files[file_type], len(file_type_map)
                )

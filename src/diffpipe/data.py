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

UNIT_WILDCARDS = {
    "lsst_": u.ABmag,
    "roman_F": u.ABmag,
    "roman_Grism": u.ABmag,
    "roman_Pirsm": u.ABmag,
}

UNIT_MAP = {
    "logmp0": u.DexUnit(u.M_sun),
    "logmp_obs": u.DexUnit(u.M_sun),
    "logmp_obs_host": u.DexUnit(u.M_sun),
    "logsm_obs": u.DexUnit(u.M_sun),
    "logssfr_obs": u.DexUnit(1 / u.year),
    "x": u.Mpc / cu.littleh,
    "y": u.Mpc / cu.littleh,
    "z": u.Mpc / cu.littleh,
    "x_host": u.Mpc / cu.littleh,
    "y_host": u.Mpc / cu.littleh,
    "z_host": u.Mpc / cu.littleh,
    "x_nfw": u.Mpc / cu.littleh,
    "y_nfw": u.Mpc / cu.littleh,
    "z_nfw": u.Mpc / cu.littleh,
    "ra": u.deg,
    "dec": u.deg,
    "ra_nfw": u.deg,
    "dec_nfw": u.deg,
    "top_host_infall_fof_halo_eigS1X": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS1Y": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS1Z": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS2X": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS2Y": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS2Z": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS3X": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS3Y": u.Mpc / cu.littleh,
    "top_host_infall_fof_halo_eigS3Z": u.Mpc / cu.littleh,
    "ra_obs": u.deg,
    "dec_obs": u.deg,
    "kappa": None,
    "shear1": None,
    "shear2": None,
}

COLUMNS_TO_SKIP = ["ra", "dec"]
COLUMN_RENAMES = {"ra_nfw": "ra", "dec_nfw": "dec"}


def process_slice(slice, step_data, index_depth, simulation):
    logger.info(f"Working on slice {slice}")
    core_files = step_data[FileType.CORE]  # required
    output_path = step_data["output_path"]  # required
    all_slices = step_data["all_slices"]  # required

    synth_core_files = step_data.get(FileType.SYNTH_CORE)  # optional

    pixels_with_data = write_files(
        slice, core_files, synth_core_files, output_path, index_depth
    )

    write_opencosmo_header(
        core_files[0],
        output_path,
        simulation,
        slice,
        all_slices,
        pixels_with_data,
        index_depth,
    )
    logger.success(f"Successfully wrote data for slice {slice}")


def write_files(slice, core_files, synth_core_files, output_path, max_level):
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
        write_single_file(max_level, files, target, file_map)
        pixels_with_data = write_indices(target, counts, max_level)
        if FileType.SYNTH_CORE in files and len(files) > 1:
            if_group = target.require_group(f"{FileType.SYNTH_CORE.value}/load/if")
            if_group.attrs["synth_cores"] = True

    # Not, we've already verified that this information is
    # consistent across all files

    target.close()
    with h5py.File(output_path) as target:
        for group_name, group in target.items():
            if group_name in ["index", "header"]:
                continue
            verify_index(group, max_level)
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


def write_single_file(max_level, source_files, file_target, file_map):
    if len(source_files) == 1:
        file_type = next(iter(source_files.keys()))
        sources = [h5py.File(s) for s in source_files[file_type]]
        for column_name in sources[0]["data"].keys():
            write_column(file_target["data"], sources, column_name, file_map[file_type])
        return
    for file_type, file_type_sources in source_files.items():
        sources = [h5py.File(s) for s in source_files[file_type]]
        for column_name in sources[0]["data"].keys():
            write_column(
                file_target[f"{file_type.value}/data"],
                sources,
                column_name,
                file_map[file_type],
            )


def write_column(output_data_group, sources, column_name, full_map):
    data = np.concat([source["data"][column_name][:] for source in sources])
    attributes = dict(sources[0]["data"][column_name].attrs)
    if column_name in COLUMNS_TO_SKIP:
        return

    output_column_name = COLUMN_RENAMES.get(column_name, column_name)

    unit = UNIT_MAP.get(column_name, None)
    if unit is None:
        for pattern, pattern_unit in UNIT_WILDCARDS.items():
            if output_column_name.startswith(pattern):
                unit = pattern_unit
                break
    try:
        bases = unit.bases
        index = bases.index(cu.littleh)
        h_power = unit.powers[index]
    except (ValueError, AttributeError):
        h_power = 0

    if h_power != 0:
        data = data / LITTLE_H**h_power
        unit = unit / LITTLE_H**h_power

    attributes["unit"] = str(unit)
    output_ds = output_data_group[output_column_name]
    output_ds[:] = data[full_map]
    output_ds.attrs.update(attributes)


def allocate_dataset_group(group, source_files, total_length):
    data_group = group.require_group("data")
    column_metadata_source = source_files[0]
    shapes = {}
    dtypes = {}
    with h5py.File(column_metadata_source) as source_f:
        for colname, col in source_f["data"].items():
            shapes[colname] = col.shape[1:]
            dtypes[colname] = col.dtype

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

import astropy.units as u
import h5py
import hdf5plugin
import numpy as np
from astropy.cosmology import units as cu

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


def write_column_attributes(target, source):
    data_group = target["data"]

    for column in source["data"].keys():
        if column in ["ra_nfw", "dec_nfw"]:
            continue
        attrs = dict(source["data"][column].attrs)
        unit = UNIT_MAP.get(column, "None")
        if unit == u.Mpc / cu.littleh:
            unit = u.Mpc

        attrs["unit"] = str(unit)
        if "units" in attrs:
            attrs.pop("units")
        data_group[column].attrs.update(attrs)


def write_files(slice, core_files, output_path, max_level):
    print(f"Working on slice {slice}")
    counts = {}

    counts["cores"] = {f: get_counts(max_level, f) for f in core_files}
    file_map = make_combined_file_map(max_level, core_files)

    target = allocate_file(output_path, core_files, len(file_map))
    write_single_file(max_level, core_files, target, file_map)
    pixels_with_data = write_index(target, counts["cores"], max_level)
    with h5py.File(core_files[0]) as attr_source:
        write_column_attributes(target, attr_source)

    version_source = core_files[0]
    with h5py.File(version_source) as f:
        version_pars = f["metadata"]["version_info"].attrs
        header = target.file.require_group("header")
        versions_group = header.require_group("diffsky_versions")
        for par, val in version_pars.items():
            versions_group.attrs[par] = val
        if meta := dict(version_source["metadata"].attrs):
            metadata_group = header.require_group("metadata")
            for par, val in meta.items():
                metadata_group[par] = val

    target.close()
    verify_index(output_path, max_level)
    return pixels_with_data


def write_index(output_file, counts, max_level):
    index_group = output_file.require_group("index")
    nside = 2**max_level
    npix = 12 * nside * nside
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
    sources = [h5py.File(s) for s in source_files]
    for column_name in sources[0]["data"].keys():
        write_column(file_target["data"], sources, column_name, file_map)


def write_column(output_data_group, sources, column_name, full_map):
    data = np.concat([source["data"][column_name][:] for source in sources])
    if column_name in ["ra", "dec"]:
        return
    elif column_name == "ra_nfw":
        column_name = "ra"
    elif column_name == "dec_nfw":
        column_name = "dec"

    unit = UNIT_MAP.get(column_name, None)
    if unit == u.Mpc / cu.littleh:
        data = data / LITTLE_H
    if "lsst" in column_name:
        unit = u.ABmag

    output_ds = output_data_group[column_name]
    output_ds[:] = data[full_map]


def allocate_file(output, files, total_length):
    output_file = h5py.File(output, "w")

    columns = set()
    file_columns = set()
    dtypes = {}

    for file in files:
        with h5py.File(file) as f:
            file_columns = set(f["data"].keys())
            columns.update(file_columns)
            dtypes = {k: col.dtype for k, col in f["data"].items()}
            shapes = {k: col.shape for k, col in f["data"].items()}
    if not file_columns == columns:
        raise ValueError("Files do not have the same columns")

    data_group = output_file.require_group("data")
    for col, dtype in dtypes.items():
        if col == "ra" or col == "dec":
            continue
        elif col == "ra_nfw":
            data_group.create_dataset(
                "ra", shape=(total_length,), dtype=dtype, compression=COMPRESSION
            )
        elif col == "dec_nfw":
            data_group.create_dataset(
                "dec", shape=(total_length,), dtype=dtype, compression=COMPRESSION
            )
        else:
            shape = (total_length,) + shapes[col][1:]

            data_group.create_dataset(
                col, shape=shape, dtype=dtype, compression=COMPRESSION
            )

    return output_file

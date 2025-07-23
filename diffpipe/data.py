from itertools import chain

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import units as cu
from numpy import random

from diffpipe.index.index import get_counts, make_file_map

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
}


class FileTarget:
    def __init__(self, file: h5py.File, counts: dict):
        self.__counts = {}
        for count_type, count in counts.items():
            pixels = np.sort(np.fromiter(count.keys(), dtype=int))
            self.__counts[count_type] = {pix: count[pix] for pix in pixels}
        self.__file = file
        self.__starts = {}

        for count_type, count in self.__counts.items():
            starts = np.cumsum(np.fromiter(count.values(), dtype=int))
            starts = np.insert(starts, 0, 0)[:-1]
            self.__starts[count_type] = {
                pix: start for pix, start in zip(count.keys(), starts)
            }

        if "data" in file.keys():
            self.__columns = {next(iter(self.__counts.keys())): file["data"].keys()}

        else:
            self.__columns = {
                name: group["data"].keys() for name, group in file.items()
            }
        self.__running_counts = {}
        for group, columns in self.__columns.items():
            self.__running_counts[group] = {
                pix: {col: 0 for col in columns} for pix in self.__counts[group].keys()
            }

    @property
    def file(self):
        return self.__file

    def close(self):
        self.__file.close()

    def write(self, group, pixel, data, one_group: bool = False):
        lengths = set(len(d) for d in data.values())
        if len(lengths) != 1:
            raise ValueError("Not all columns have the same length!")

        col_length = lengths.pop()
        running_counts = self.__running_counts[group][pixel]

        for column, data in data.items():
            if column in ["ra", "dec"]:
                continue
            elif column == "ra_nfw":
                column = "ra"
            elif column == "dec_nfw":
                column = "dec"
            start = self.__starts[group][pixel]
            rc = running_counts[column]
            if one_group:
                data_group = self.__file["data"]
            else:
                data_group = self.__file[group]["data"]

            column_ds = data_group[column]

            unit = UNIT_MAP.get(column, None)
            if unit == u.Mpc / cu.littleh:
                data = data / LITTLE_H

            column_ds[start + rc : start + rc + col_length] = data
            running_counts[column] = rc + col_length

        self.__running_counts[group][pixel] = running_counts

    def write_index(self, max_level, one_group: bool = False):
        for group, group_counts in self.__counts.items():
            if len(self.__counts.values()) == 1:
                index_group = self.__file.require_group("index")
            else:
                index_group = self.__file[group].require_group("index")
            nside = 2**max_level
            npix = 12 * nside * nside
            starts = np.zeros(npix, dtype=np.int32)
            sizes = np.zeros(npix, dtype=np.int32)
            for pixel in group_counts.keys():
                sizes[pixel] = group_counts[pixel]
                starts[pixel] = self.__starts[group][pixel]

            for level in range(max_level, -1, -1):
                level_group = index_group.require_group(f"level_{level}")
                level_group.create_dataset("start", data=starts)
                level_group.create_dataset("size", data=sizes)
                sizes = sizes.reshape(-1, 4).sum(axis=1)
                starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]

    def write_column_attributes(self, key, source):
        if len(self.__counts) == 1:
            data_group = self.__file["data"]
        else:
            data_group = self.__file[key]["data"]

        for column in source.keys():
            if column in ["ra_nfw", "dec_nfw"]:
                continue
            attrs = dict(source[column].attrs)
            unit = UNIT_MAP.get(column, "None")
            if unit == u.Mpc / cu.littleh:
                unit = u.Mpc

            attrs["unit"] = str(unit)
            if "units" in attrs:
                attrs.pop("units")
            data_group[column].attrs.update(attrs)


def write_files(slice, files, max_level):
    print(f"Working on slice {slice}")
    output = files.pop("output")
    counts = {}
    for file_type, fs in files.items():
        counts[file_type] = {f: get_counts(max_level, f) for f in fs}

    one_group = len(files.keys()) == 1
    target = allocate_file(output, files, counts, one_group)
    for file_type, fs in files.items():
        for f in fs:
            write_single_file(max_level, file_type, f, target, one_group)

    target.write_index(max_level)
    for key, fs in files.items():
        with h5py.File(fs[0]) as f:
            target.write_column_attributes(key, f["data"])

    if "synthetic_cores" in target.file.keys():
        load_group = target.file["synthetic_cores"].require_group("load/if")
        load_group.attrs["synth_cores"] = True

    version_source = files["cores"][0]
    with h5py.File(version_source) as f:
        version_pars = f["metadata"]["version_info"].attrs
        header = target.file.require_group("header")
        versions_group = header.require_group("diffsky_versions")
        for par, val in version_pars.items():
            versions_group.attrs[par] = val

    target.close()
    # print("Verifying file")
    # verify(files, output)


def verify(input_files, output_file):
    output = h5py.File(output_file)
    datasets = list(output["data"].keys())
    to_check = random.choice(datasets, 10)
    files = [h5py.File(f) for f in input_files]
    rows_to_check = np.unique(random.randint(0, len(output["data"]["core_tag"]), 100))
    output_rows = {}
    for column in to_check:
        read = np.concat([f["data"][column][:] for f in files])
        written = output["data"][column][:]
        read_values, read_counts = np.unique(read, return_counts=True)
        written_values, written_counts = np.unique(written, return_counts=True)

        output_rows[column] = written[rows_to_check]

        assert np.all(read_values == written_values)
        assert np.all(read_counts == written_counts)

    rows = np.vstack([r for r in output_rows.values()]).T
    read_data = {}
    for column in output_rows.keys():
        read_data[column] = np.concat([f["data"][column][:] for f in files])
    read_data = np.vstack([r for r in read_data.values()]).T
    for row in rows:
        assert np.any(np.all(row == read_data, axis=1))


def write_single_file(
    max_level, file_type, source_file, file_target, one_group: bool = False
):
    maps = make_file_map(max_level, source_file)
    with h5py.File(source_file, "r") as f:
        for key, ds in f["data"].items():
            data = ds[:]

            for pixel, m in maps.items():
                file_target.write(file_type, pixel, {key: data[m]}, one_group)


def combine_counts(counts):
    unique_keys = set(chain.from_iterable(c.keys() for c in counts.values()))
    totals = {}
    for key in unique_keys:
        total = sum(c.get(key, 0) for c in counts.values())
        totals[key] = total
    return totals


def allocate_file(output, files, counts, one_group: bool = False):
    print(f"Allocating file {output}")
    output_file = h5py.File(output, "w")
    totals = {}
    for count_type, count in counts.items():
        count_totals = combine_counts(count)
        totals[count_type] = count_totals
        columns = set()
        file_columns = set()
        dtypes = {}
        for file in files[count_type]:
            with h5py.File(file) as f:
                file_columns = set(f["data"].keys())
                columns.update(file_columns)
                dtypes = {k: col.dtype for k, col in f["data"].items()}
        if not file_columns == columns:
            raise ValueError("Files do not have the same columns")

        total_length = sum(count_totals.values())
        if one_group:
            data_group = output_file.require_group("data")
        else:
            type_group = output_file.require_group(count_type)
            data_group = type_group.require_group("data")
        for col, dtype in dtypes.items():
            if col == "ra" or col == "dec":
                continue
            elif col == "ra_nfw":
                data_group.create_dataset("ra", shape=(total_length,), dtype=dtype)
            elif col == "dec_nfw":
                data_group.create_dataset("dec", shape=(total_length,), dtype=dtype)
            else:
                data_group.create_dataset(col, shape=(total_length,), dtype=dtype)
    print(f"Total length {total_length}")

    return FileTarget(output_file, totals)

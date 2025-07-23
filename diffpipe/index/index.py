from itertools import chain

import h5py
import numpy as np
from healpy.pixelfunc import ang2pix


def get_counts(max_level, file):
    file = h5py.File(file)
    ra = file["data"]["ra_nfw"][:]
    dec = file["data"]["dec_nfw"][:]
    nside = 2**max_level

    idxs = ang2pix(nside, ra, dec, lonlat=True, nest=True)

    indices, counts = np.unique(idxs, return_counts=True)
    return {i: c for i, c in zip(indices, counts)}


def get_arrangement(max_level, files):
    maps = {}
    for file in files:
        with h5py.File(file) as f:
            ra = f["data"]["ra_nfw"][:]
            dec = f["data"]["dec_nfw"][:]
            maps.update({file: make_map(max_level, ra, dec)})
    index = make_index(max_level, *maps.values())
    return maps, index


def make_file_map(max_level: int, file):
    with h5py.File(file) as f:
        ra = f["data"]["ra_nfw"][:]
        dec = f["data"]["dec_nfw"][:]
        return make_map(max_level, ra, dec)


def make_map(max_level: int, ra: np.ndarray, dec: np.ndarray):
    nside = 2**max_level
    idxs = ang2pix(nside, ra, dec, lonlat=True, nest=True)

    indices, counts = np.unique(idxs, return_counts=True)
    count_totals = np.cumsum(counts)[:-1]

    sort = np.argsort(idxs)
    sort = np.split(sort, count_totals)
    return {idx: slice for idx, slice in zip(indices, sort)}


def make_index(max_level: int, *maps: dict):
    nside = 2**max_level
    spatial_index = np.zeros(12 * nside * nside, dtype=np.uint32)
    unique_idxs = set(chain.from_iterable(m.keys() for m in maps))
    for idx in unique_idxs:
        spatial_index[idx] = sum(len(m.get(idx, [])) for m in maps)

    return spatial_index

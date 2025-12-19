from itertools import chain

import h5py
import numpy as np
from healpy.pixelfunc import ang2pix


def verify_index(group, max_level):
    ra = group["data"]["ra"][:]
    dec = group["data"]["dec"][:]
    pixel_assignments = ang2pix(2**max_level, ra, dec, lonlat=True, nest=True)
    size = group["index"][f"level_{max_level}"]["size"][:]
    pixels_to_check = np.where(size > 0)[0]
    pixels, sizes = np.unique(pixel_assignments, return_counts=True)
    assert np.all(pixels == pixels_to_check)
    assert np.all(sizes == size[pixels_to_check])
    assert np.all(np.sort(pixel_assignments) == pixel_assignments)


def get_counts(max_level, file):
    file = h5py.File(file)
    ra = file["data"]["ra_nfw"][:]
    dec = file["data"]["dec_nfw"][:]
    nside = 2**max_level

    idxs = ang2pix(nside, ra, dec, lonlat=True, nest=True)

    indices, counts = np.unique(idxs, return_counts=True)
    return {i: c for i, c in zip(indices, counts)}


def combine_counts(counts):
    unique_keys = set(chain.from_iterable(c.keys() for c in counts.values()))
    totals = {}
    for key in unique_keys:
        total = sum(c.get(key, 0) for c in counts.values())
        totals[key] = total
    return totals


def make_combined_file_map(max_level: int, files: list):
    pixel_assignments = [get_pixels(max_level, f) for f in files]

    pixel_assignments = np.concat(pixel_assignments)
    file_maps = np.argsort(pixel_assignments)

    return file_maps


def get_pixels(max_level: int, file):
    with h5py.File(file) as f:
        ra = f["data"]["ra_nfw"][:]
        dec = f["data"]["dec_nfw"][:]
    nside = 2**max_level
    return ang2pix(nside, ra, dec, lonlat=True, nest=True)

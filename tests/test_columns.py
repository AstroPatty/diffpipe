import re
from collections import defaultdict
from typing import Any

import h5py
import numpy as np
import opencosmo as oc
from numpy import dtype

KNOWN_NEW = {
    "gal_id",
}  # Columns that only appear in one catalog
KNOWN_RENAMES = {"dec_nfw": "dec", "ra_nfw": "ra"}
KNOWN_DIFFS = {"ra": ("description",), "dec": ("description",)}

# Columns whose values are intentionally transformed during conversion rather
# than copied verbatim. ``top_host_idx`` is a self-referential row index that
# the pipeline reindexes to the new row ordering, so it is not value-for-value
# identical to the diffsky source and cannot be compared directly.
KNOWN_TRANSFORMED = {
    "top_host_idx",
}

# The original "ra"/"dec" columns are dropped from the diffsky catalog; the
# "ra_nfw"/"dec_nfw" columns are then renamed to take their place.
DROPPED = set(KNOWN_RENAMES.values())


def _get_column_metadata(group: h5py.Group | h5py.File) -> dict[str, dtype]:
    dtypes = {}
    attrs = {}
    for key, column in group.items():
        assert isinstance(column, h5py.Dataset)
        dtypes[key] = column.dtype
        attrs[key] = dict(column.attrs)
    return dtypes, attrs


def _verify_consistency(d1: dict[str, Any], d2: dict[str, Any]):
    all_known_keys = set(d1.keys()).intersection(d2.keys())
    for key in all_known_keys:
        v1 = d1.get(key)
        v2 = d2.get(key)
        if (v1 is None or v2 is None) and key in KNOWN_NEW:
            continue
        if v1 == v2:
            continue
        _do_diff_check(key, v1, v2)


def _do_diff_check(name: str, v1, v2):
    if name not in KNOWN_DIFFS:
        raise ValueError(f"Found a difference in column {name}: {v1} -> {v2}")
    diff = KNOWN_DIFFS.get(name)
    if not diff:
        return
    all_keys = set(v1.keys()).union(v2.keys())
    for key in all_keys:
        subv1 = v1.get(key)
        subv2 = v2.get(key)
        if subv1 != subv2 and key not in diff:
            raise ValueError(
                f"Detected a difference in attributes for column {name}: {subv1} -> {subv2}"
            )


def _partition_by_step(files):
    files = [str(f) for f in files]
    output = defaultdict(list)
    for file in files:
        match = re.search(r"\d+", file)
        assert match is not None
        step = int(match.group())
        output[step].append(file)

    return dict(output)


def test_column_metadata(input_files, output_files):
    """Both catalog directories exist and contain HDF5 files."""

    # diffpipe already asserts during its run that matching columns have
    # matching columns across input files have matching metadata, so we
    # only need to check one input file
    with h5py.File(input_files[0]) as f:
        dtypes, attrs = _get_column_metadata(f["data"])

    for ofile in output_files:
        with h5py.File(ofile) as f:
            dtypes_cores, attrs_cores = _get_column_metadata(f["cores"]["data"])
            dtypes_synth, attrs_synth = _get_column_metadata(f["cores"]["data"])

        _verify_consistency(dtypes, dtypes_cores)
        _verify_consistency(dtypes, dtypes_synth)
        _verify_consistency(attrs, attrs_cores)
        _verify_consistency(attrs, attrs_synth)


def test_data_consistency(input_files, output_files):
    input_files_by_step = _partition_by_step(input_files)
    output_files_by_step = _partition_by_step(output_files)
    if not set(input_files_by_step.keys()) == set(output_files_by_step.keys()):
        raise ValueError("diffsky and opencosmo data don't have the same steps!")

    for step, opencosmo_files in output_files_by_step.items():
        assert len(opencosmo_files) == 1
        diffsky_files = input_files_by_step[step]
        dataset = oc.open(opencosmo_files[0], synth_cores=True)
        mapping = _get_mapping(dataset, diffsky_files)
        _verify_all_columns(dataset, diffsky_files, mapping)


def test_top_host_idx_consistency(input_files, output_files):
    """``top_host_idx`` points at another row (an object's "top host").

    It is the one column that is not copied verbatim (see ``KNOWN_TRANSFORMED``):
    the pipeline reindexes it to the reordered rows. Rather than compare values
    directly, verify the pointer still references the same physical object after
    conversion. Given the row mapping (``opencosmo_row[i] == diffsky_row[m[i]]``),
    a correctly reindexed pointer satisfies ``m[opencosmo_ptr] == diffsky_ptr[m]``.
    """
    input_files_by_step = _partition_by_step(input_files)
    output_files_by_step = _partition_by_step(output_files)
    if not set(input_files_by_step.keys()) == set(output_files_by_step.keys()):
        raise ValueError("diffsky and opencosmo data don't have the same steps!")

    for step, opencosmo_files in output_files_by_step.items():
        assert len(opencosmo_files) == 1
        diffsky_files = input_files_by_step[step]
        dataset = oc.open(opencosmo_files[0], synth_cores=True)
        mapping = _get_mapping(dataset, diffsky_files)

        # opencosmo numbers top_host_idx as a global row index into the combined
        # catalog, while each diffsky file numbers its rows from zero. Shift the
        # per-file diffsky indices by each file's start offset so both sides
        # reference the same concatenated row space that ``mapping`` was built on.
        diffsky_ptr = _read_diffsky_index(diffsky_files, "top_host_idx")

        opencosmo_ptr = _read_opencosmo_column(dataset, "top_host_idx")
        if isinstance(opencosmo_ptr, dict):
            opencosmo_ptr = opencosmo_ptr["top_host_idx"]

        assert np.array_equal(mapping[opencosmo_ptr], diffsky_ptr[mapping]), (
            f"top_host_idx is not consistent under the row mapping for step {step}"
        )


def _read_diffsky_column(diffsky_files, column_name):
    """Concatenate a single column across the diffsky input files."""
    chunks = []
    for file in diffsky_files:
        with h5py.File(file) as f:
            chunks.append(f["data"][column_name][:])
    return np.concatenate(chunks)


def _read_diffsky_index(diffsky_files, column_name):
    """Concatenate a group-local row-index column across the diffsky files.

    Each file numbers its rows from zero, so add every file's start offset to
    turn the values into indices into the concatenated rows.
    """
    chunks = []
    offset = 0
    for file in diffsky_files:
        with h5py.File(file) as f:
            values = f["data"][column_name][:]
        chunks.append(values + offset)
        offset += len(values)
    return np.concatenate(chunks)


def _read_opencosmo_column(dataset, column_name):
    """Read a single column from the opencosmo dataset as a numpy array."""
    return dataset.select(column_name).get_data("numpy")
    # A single-column select usually returns a bare ndarray, but some columns
    # come back as a mapping keyed by the column name.


def _verify_all_columns(opencosmo_dataset, diffsky_files, mapping):
    """Every diffsky column is copied value-for-value into the opencosmo data.

    ``mapping`` reorders diffsky rows into opencosmo row order (see
    ``_get_mapping``), i.e. ``opencosmo_row[i] == diffsky_row[mapping[i]]``.
    """
    with h5py.File(diffsky_files[0]) as f:
        diffsky_columns = list(f["data"].keys())

    for diffsky_name in diffsky_columns:
        # The original ra/dec columns are dropped and replaced by the renamed
        # ra_nfw/dec_nfw columns, so skip the originals.
        if diffsky_name in DROPPED:
            continue
        # Transformed index columns are not value-for-value identical.
        if diffsky_name in KNOWN_TRANSFORMED:
            continue

        opencosmo_name = KNOWN_RENAMES.get(diffsky_name, diffsky_name)

        diffsky_values = _read_diffsky_column(diffsky_files, diffsky_name)
        opencosmo_values = _read_opencosmo_column(opencosmo_dataset, opencosmo_name)

        assert opencosmo_values.shape == diffsky_values.shape, (
            f"Shape mismatch for column {diffsky_name!r} -> {opencosmo_name!r}: "
            f"{diffsky_values.shape} != {opencosmo_values.shape}"
        )
        assert np.array_equal(opencosmo_values, diffsky_values[mapping]), (
            f"Column values differ between diffsky {diffsky_name!r} and "
            f"opencosmo {opencosmo_name!r}"
        )


def _get_mapping(dataset, diffsky_files):
    opencosmo_coordinates = dataset.select("ra", "dec").get_data("numpy")
    opencosmo_coordinates = np.column_stack(
        (opencosmo_coordinates["ra"], opencosmo_coordinates["dec"])
    )

    diffsky_ras = []
    diffsky_decs = []
    for file in diffsky_files:
        with h5py.File(file) as f:
            diffsky_ras.append(f["data"]["ra_nfw"][:])
            diffsky_decs.append(f["data"]["dec_nfw"][:])
    diffsky_ras = np.concatenate(diffsky_ras)
    diffsky_decs = np.concatenate(diffsky_decs)
    diffsky_coordinates = np.column_stack((diffsky_ras, diffsky_decs))

    assert diffsky_coordinates.shape == opencosmo_coordinates.shape

    ia = np.lexsort((opencosmo_coordinates[:, 1], opencosmo_coordinates[:, 0]))
    ib = np.lexsort((diffsky_coordinates[:, 1], diffsky_coordinates[:, 0]))

    mapping = np.empty(len(opencosmo_coordinates), dtype=int)
    mapping[ia] = ib
    assert len(np.unique(mapping)) == len(mapping)
    assert np.all(opencosmo_coordinates == diffsky_coordinates[mapping])
    return mapping

    # Get indices where the rows are exactly equal

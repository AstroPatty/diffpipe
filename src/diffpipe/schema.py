"""
Verification that every input file has the same columns and metadata.

Checking every file up front means opening each one before any work starts, which
can take hours on a network file system. Instead, each slice checks its own files
once they are open for conversion. The first slice to get there writes a schema
(the columns and metadata of its files) to a file shared by the run, and every
other slice is checked against it.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from diffpipe import BadInputError
from diffpipe.header import ALLOWED_METADATA_DIFFS, EXPECTED_METDATA_ATTRIBUTE_GROUPS

SCHEMA_FILE_NAME = ".diffpipe-schema.json"


@dataclass
class Schema:
    # column name -> (shape beyond the 0th axis, dtype, attributes)
    columns: dict[str, tuple[tuple, np.dtype, dict]]
    # metadata group -> attributes
    metadata: dict[str, dict]


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


def read_schema(f: h5py.File) -> Schema:
    """
    Read the schema of a catalog file.
    """
    if (
        "data" not in f
        or "metadata" not in f
        or "mock_version_name" not in f["metadata"].attrs
    ):
        raise BadInputError(f"{f.filename} is not a diffsky catalog file")
    try:
        columns = get_columns_in_group(f["data"])
    except ValueError as e:
        raise BadInputError(f"{f.filename}: {e}")

    metadata = {}
    for group_name in EXPECTED_METDATA_ATTRIBUTE_GROUPS:
        if group_name not in f:
            raise BadInputError(
                f"{f.filename} is missing expected metadata group {group_name}"
            )
        metadata[group_name] = dict(f[group_name].attrs)

    return Schema(
        columns={
            name: (col.shape[1:], col.dtype, dict(col.attrs))
            for name, col in columns.items()
        },
        metadata=metadata,
    )


def encode_value(value):
    """
    Encode an attribute value as JSON, tagging numpy and bytes values so they
    decode to exactly what h5py would read from a file.
    """
    if isinstance(value, np.ndarray):
        return {
            "ndarray": [encode_value(v) for v in value.ravel().tolist()],
            "shape": value.shape,
            "dtype": value.dtype.str,
        }
    if isinstance(value, np.generic):
        return {"scalar": encode_value(value.item()), "dtype": value.dtype.str}
    if isinstance(value, bytes):
        return {"bytes": value.decode("latin-1")}
    return value


def decode_value(value):
    if not isinstance(value, dict):
        return value
    if "ndarray" in value:
        values = [decode_value(v) for v in value["ndarray"]]
        return np.array(values, dtype=value["dtype"]).reshape(value["shape"])
    if "scalar" in value:
        return np.dtype(value["dtype"]).type(decode_value(value["scalar"]))
    return value["bytes"].encode("latin-1")


def encode_attrs(attrs: dict) -> dict:
    return {key: encode_value(value) for key, value in attrs.items()}


def decode_attrs(attrs: dict) -> dict:
    return {key: decode_value(value) for key, value in attrs.items()}


def write_schema(schema: Schema, path: Path):
    columns = {
        name: {"shape": shape, "dtype": dtype.str, "attrs": encode_attrs(attrs)}
        for name, (shape, dtype, attrs) in schema.columns.items()
    }
    metadata = {
        group_name: encode_attrs(attrs) for group_name, attrs in schema.metadata.items()
    }
    path.write_text(json.dumps({"columns": columns, "metadata": metadata}))


def publish_schema(schema: Schema, path: Path) -> bool:
    """
    Write the schema to path, unless another slice already has. Returns whether
    this one was written. The schema is written under a temporary name and linked
    into place, which fails if the file exists, so a reader never sees a partial
    schema and two slices can't both write one.
    """
    partial_path = path.with_name(f"{path.name}.{os.getpid()}.partial")
    write_schema(schema, partial_path)
    try:
        os.link(partial_path, path)
        return True
    except FileExistsError:
        return False
    finally:
        partial_path.unlink()


def load_schema(path: Path) -> Schema:
    contents = json.loads(path.read_bytes())
    return Schema(
        columns={
            name: (
                tuple(col["shape"]),
                np.dtype(col["dtype"]),
                decode_attrs(col["attrs"]),
            )
            for name, col in contents["columns"].items()
        },
        metadata={
            group_name: decode_attrs(attrs)
            for group_name, attrs in contents["metadata"].items()
        },
    )


def values_equal(a, b) -> bool:
    return np.array_equal(a, b) if isinstance(a, np.ndarray) else bool(a == b)


def attrs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)


def verify_schema(reference: Schema, schema: Schema, filename: str):
    """
    Verify a file's schema against the reference:
    1. The metadata is the same, ignoring entries that differ in a normal run
    2. The columns are the same, and for each column
       a. the shapes are identical after the 0th axis
       b. the attributes are identical
       c. the dtypes have the same base type (numpy promotes the precision later)
    """
    for group_name, reference_attrs in reference.metadata.items():
        attrs = schema.metadata[group_name]
        for key, value in reference_attrs.items():
            if key not in attrs:
                raise BadInputError(
                    f"{filename} is missing metadata entry {group_name}/{key}"
                )
            if key in ALLOWED_METADATA_DIFFS.get(group_name, []):
                continue
            if not values_equal(value, attrs[key]):
                raise BadInputError(
                    f"{filename} has different metadata for entry {group_name}/{key}"
                )

    if schema.columns.keys() != reference.columns.keys():
        missing = reference.columns.keys() - schema.columns.keys()
        extra = schema.columns.keys() - reference.columns.keys()
        raise BadInputError(
            f"{filename} does not have the same columns as other files "
            f"(missing {sorted(missing)}, extra {sorted(extra)})"
        )
    for name, (rshape, rdtype, rattrs) in reference.columns.items():
        shape, dtype, attrs = schema.columns[name]
        if shape != rshape:
            raise BadInputError(f"{filename} has a different shape for column {name}")
        if not attrs_equal(attrs, rattrs):
            raise BadInputError(
                f"{filename} has different attributes for column {name}"
            )
        # Columns may have different precision (numpy will automatically promote
        # later), but we are avoiding situations where we have eg. int and float
        if np.promote_types(rdtype, dtype) not in [rdtype, dtype]:
            raise BadInputError(
                f"{filename} has an incompatible dtype for column {name} "
                f"({dtype}, other files have {rdtype})"
            )


def verify_slice_files(slice, files: list[h5py.File], schema_path: Path):
    """
    Verify a slice's (open) input files against the schema for the run, writing
    the schema from these files if no other slice has yet.
    """
    schemas = {f.filename: read_schema(f) for f in files}
    try:
        reference = load_schema(schema_path)
    except FileNotFoundError:
        reference = next(iter(schemas.values()))
        for filename, schema in schemas.items():
            verify_schema(reference, schema, filename)
        if publish_schema(reference, schema_path):
            logger.info(f"Slice {slice} wrote the schema for this run")
            return
        reference = load_schema(schema_path)

    for filename, schema in schemas.items():
        verify_schema(reference, schema, filename)

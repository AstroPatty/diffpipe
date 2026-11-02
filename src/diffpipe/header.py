from importlib.resources import files as resource_files

import h5py
import numpy as np
from loguru import logger

try:
    from opencosmo.spatial.models import HealpixRegionModel
except ImportError:
    from opencosmo.spatial.models import HealPixRegionModel as HealpixRegionModel

from diffpipe import BadInputError

STEPS = {
    "LastJourney": {
        "Z_INI": 200.0,
        "Z_FIN": 0.0,
        "N_STEPS": 500,
    }
}


EXPECTED_METDATA_ATTRIBUTE_GROUPS = [  # We will verify these are consistent across files
    "metadata",
    "metadata/cosmology",
    "metadata/nbody_info",
    "metadata/software_version_info",
]

ALLOWED_METADATA_DIFFS = {"metadata": ["creation_date", "mock_version_name"]}


METADATA_ATTRIBUTE_GROUPS_TO_COPY = {  # These will be copied into the OpenCosmo Header
    "metadata": "catalog_info",
    "metadata/software_version_info": "diffsky_versions",
}


def get_step_zs(simulation_name):
    spec = STEPS[simulation_name]
    steps_a = np.linspace(
        1 / (1 + spec["Z_INI"]), 1 / (1 + spec["Z_FIN"]), spec["N_STEPS"] + 1
    )[1:]
    steps_z = 1 / steps_a - 1
    return steps_z


def copy_required_header_attributes(source: h5py.File, output: h5py.File):
    for input_name, output_name in METADATA_ATTRIBUTE_GROUPS_TO_COPY.items():
        group = output.require_group(f"header/{output_name}")
        group.attrs.update(dict(source[input_name].attrs))


def get_simulation_header_data(simulation_name: str):
    header_path = resource_files("diffpipe") / "headers" / f"{simulation_name}.hdf5"
    if not header_path.exists():
        logger.critical(
            f"Could not find header information for simulation {simulation_name}"
        )
        raise BadInputError

    with h5py.File(header_path) as source:
        data = __recurse_header(source, "header")
    return data


def __recurse_header(source: h5py.File, group_name: str):
    output = {}
    group_data = dict(source[group_name].attrs)
    if group_data:
        output[group_name] = group_data
    for subgroup_name, subgroup in source[group_name].items():
        subgroup_path = f"{group_name}/{subgroup_name}"
        output |= __recurse_header(source, subgroup_path)
    return output


def write_opencosmo_header(
    source: h5py.File,
    f_output: h5py.File,
    simulation_name: str,
    step: int,
    all_steps: set[int],
    pixels_with_data: np.ndarray,
    index_depth: int,
    z_phot_table: np.ndarray,
):
    copy_required_header_attributes(source, f_output)
    additional_header_data = get_simulation_header_data(simulation_name)
    simulation_steps = get_step_zs(simulation_name)

    FILE_PARS = {
        "data_type": "diffsky_fits",
        "is_lightcone": True,
        "redshift": simulation_steps[step],
        "step": step,
        "unit_convention": "comoving",
        "origin": "HACC",
    }
    HealpixRegionModel(pixels=pixels_with_data, nside=2**index_depth).model_dump()
    # for key, value in region_model.items():
    #    FILE_PARS[f"region_{key}"] = value
    all_steps = np.fromiter(all_steps, dtype=int)
    all_steps.sort()
    idx = np.where(all_steps == step)[0][0]
    redshift_high = simulation_steps[step]
    try:
        redshift_low = simulation_steps[all_steps[idx + 1]]
    except IndexError:
        redshift_low = 0.0

    for group_path, group_data in additional_header_data.items():
        group = f_output.require_group(group_path)
        group.attrs.update(group_data)

    file_group = f_output.require_group("header/file")
    for key, val in FILE_PARS.items():
        file_group.attrs[key] = val
    lightcone_group = f_output.require_group("header/lightcone")
    lightcone_group.attrs["z_range"] = [redshift_low, redshift_high]
    f_output["header"]["catalog_info"].create_dataset("zphot_table", data=z_phot_table)

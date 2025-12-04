import h5py
import numpy as np
from opencosmo.spatial.models import HealPixRegionModel

LASTJOURNEY_NSTEPS = 500
I_SEED = 10041972
NG = 10752
NP = 10752
RL = 3400.0
Z_INI = 200.0
Z_FIN = 0.0
N_STEPS = 500
N_SUB = 5
OL = 2.0
ANALYSIS_OL = 8.0
RSM = 0.01
RCB_TREE_PPN = 128

lj_steps_a = np.linspace(1 / (1 + Z_INI), 1 / (1 + Z_FIN), N_STEPS + 1)[1:]
lj_steps = 1 / lj_steps_a - 1


def copy_header(
    source: str,
    dest: str,
    step: int,
    all_steps: set[int],
    pixels_with_data: np.ndarray,
    nside: int,
):
    FILE_PARS = {
        "data_type": "diffsky_fits",
        "is_lightcone": True,
        "redshift": lj_steps[step],
        "step": step,
        "unit_convention": "comoving",
        "origin": "HACC",
    }
    region_model = HealPixRegionModel(pixels=pixels_with_data, nside=nside).model_dump()
    for key, value in region_model:
        FILE_PARS[f"region_{key}"] = value
    all_steps = np.fromiter(all_steps, dtype=int)
    all_steps.sort()
    idx = np.where(all_steps == step)[0][0]
    redshift_high = lj_steps[step]
    try:
        redshift_low = lj_steps[all_steps[idx + 1]]
    except IndexError:
        redshift_low = 0.0

    with h5py.File(dest, "a") as f_dest:
        with h5py.File(source) as f_source:
            for key, val in f_source["header"].items():
                f_source.copy(val, f_dest["header"])
        file_group = f_dest.require_group("header/file")
        for key, val in FILE_PARS.items():
            file_group.attrs[key] = val
        lightcone_group = f_dest.require_group("header/lightcone")
        lightcone_group.attrs["z_range"] = [redshift_low, redshift_high]

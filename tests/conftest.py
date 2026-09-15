"""Shared pytest fixtures for the diffpipe property test suite.

The location of the test data is provided via the ``DIFFPIPE_TEST_DATA_DIR``
environment variable. That directory is expected to contain two
subdirectories, each holding one of the two HDF5 schemas diffpipe converts
between:

    $DIFFPIPE_TEST_DATA_DIR/
        input/    # diffsky galaxy catalog (conversion input)
        output/   # OpenCosmo catalog (conversion output)

In CI these are populated by downloading the artifacts emitted by the
mock-generation job. When the environment variable is not set (e.g. a local
run without data) the dependent tests are skipped rather than failed.
"""

import os
from pathlib import Path

import pytest

ENV_VAR = "DIFFPIPE_TEST_DATA_DIR"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Root directory containing the ``input`` and ``output`` catalogs."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raw = Path(__file__).parent / "test_data"
    path = Path(raw)
    if not path.is_dir():
        pytest.fail(f"{ENV_VAR}={raw!r} does not point to an existing directory.")
    return path


@pytest.fixture(scope="session")
def input_dir(data_dir: Path) -> Path:
    """Directory holding the diffsky input catalog (conversion source)."""
    return data_dir / "input"


@pytest.fixture(scope="session")
def output_dir(data_dir: Path) -> Path:
    """Directory holding the OpenCosmo output catalog (conversion result)."""
    return data_dir / "output"


@pytest.fixture(scope="session")
def input_files(input_dir: Path) -> list[Path]:
    """Sorted list of HDF5 files in the input catalog."""
    return sorted(input_dir.glob("lc_cores*.hdf5"))


@pytest.fixture(scope="session")
def output_files(output_dir: Path) -> list[Path]:
    """Sorted list of HDF5 files in the output catalog."""
    return sorted(output_dir.glob("lc_cores*.hdf5"))

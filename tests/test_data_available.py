"""Smoke test verifying the converted catalog data is present.

This is the scaffold entry point for the property test suite: it confirms that
both schemas (diffsky input and OpenCosmo output) are reachable via the
fixtures before more detailed property tests build on top of them.
"""


def test_data_available(input_dir, output_dir, input_files, output_files):
    """Both catalog directories exist and contain HDF5 files."""
    assert input_dir.is_dir(), f"input catalog directory missing: {input_dir}"
    assert output_dir.is_dir(), f"output catalog directory missing: {output_dir}"

    assert input_files, f"no HDF5 files found in input catalog: {input_dir}"
    assert output_files, f"no HDF5 files found in output catalog: {output_dir}"

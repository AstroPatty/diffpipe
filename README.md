# Diffpipe

Diffpipe converts raw data from a Diffsky synthetic galaxy catalog run into the OpenCosmo format. 

# Installation

```bash
pip install git+https://github.com/AstroPatty/diffpipe.git
```

# Usage

This tool exposes a single command line script, `diffpipe run`. See `diffpipe run --help` for usage details. It produces nice logs you can check, and should always fail with an easy-to-understand error.

# Implemented Checks

This package does extensive checking on the incoming data before performing any actual work. Currently implemented (in order of appearance):

1. If working with both cores and synthetic cores, ensure we have both types of files for every redshift slice
2. If working with both cores and synthetic cores, check each redshift slice and ensure we have both cores and synthetic cores for every pixel included in that slice.
3. Ensure that metadata is consistent across all files, ignoring elements that can be different in a normal run (e.g. "file-creation-date")
4. Ensure that all files have the same set of columns
5. For each column, ensure that shapes are consistent (identical beyond the 0th dimension) across all files
6. For each column, ensure that metdata (e.g. description and unit) is consistent across all files 
7. For each column, ensure that data types are consistent across all files. Data types do not need to have the same precision, but should have the same base type (e.g. float32 and float64 is allowed, float32 and int32 is not). Data in the output file is always promoted to the highest precision of all inputs.




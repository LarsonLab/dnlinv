# Deep Non-linear Inversion (DNLINV)

This directory contains the source code for reconstructing images using deep non-linear inversion (DNLINV).

To run, simply do:
`python reconstruct.py <input_file> <output_directory> <parameters>`

By default, `<input_file>` is an HDF5 file with format following the fastMRI project knee challenge. Data in 
BART CFL format can be reconstructed by passing the argument `--h5_type bart`. And data in the ISMRMRD format 
in raw data from MRSRL files (e.g., from mridata.org) can be reconstructed by passing the argument 
`--h5_type mrsrl-ismrmrd`

Print the arguments with `python reconstruct.py -h`




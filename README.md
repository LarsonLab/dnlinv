# Deep Non-linear Inversion Reproducibility Scripts

This repository contains scripts for reproducing the experiments for the paper: ...

The reconstruction code for deep non-linear inversion is inside the `dnlinv` directory.

The scripts for reproducing the experiments in the paper is inside the `reproducibility` directory.

# Software requirements

1. Operating system: the code has been tested on Windows 10 64-bit using Cygwin, Ubuntu 18.04, and RHEL 7 operating system environments.
2. BART Toolbox v0.6.00
3. Miniconda with Python 3 (see below)

# Hardware requirements

All DNLINV reconstruction code requires an NVIDIA GPU with at least 8GB of GPU memory. At least 32GB of RAM is recommended.



# Setup instructions for reproducible experiments

1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

2. Set up conda environments
    1. `conda env create -n dnlinv -f dnlinv_environment.yml python=3.6`
    2. `conda env create -n fastMRI -f fastMRI_environment.yml pyhon=3.8`
3. Install BART toolbox v0.6.00 (recommended): https://github.com/mrirecon/bart/releases/tag/v0.6.00
4. Set up paths to the DNLINV, fastMRI, and BART repositories and executables
    1. Edit the file `setup_paths.py` to assign the correct paths to `DNLINV_PATH`, `FASTMRI_PATH`, and (bart) `TOOLBOX_PATH`
    2. Run the command `source setup_paths.py`
    3. If you plan on using the provided SLURM-compatible job submission scripts, append sourcing the `setup_paths.py` to
       the SLURM script or to your `.bashrc` file. E.g. `source <path_to_this_repo>/setup_paths.py`

5. Run scripts inside each directory in `reproducibility`. Follow the directory-specific instructions.



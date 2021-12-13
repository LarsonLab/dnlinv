#!/bin/bash
conda activate dnlinv

set -eo pipefail

./0_reco_dnlinv.sh
./1_reco_enlive.sh
conda activate fastMRI
./2_reco_fastMRI_unet.sh
conda activate dnlinv
./3_get_measurements.sh
./4_create_us_fig.sh

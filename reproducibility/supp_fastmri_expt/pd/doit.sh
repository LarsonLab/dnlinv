#!/bin/bash
conda activate dnlinv

set -eo pipefail

./0_reco_dnlinv.sh
./1_reco_enlive.sh
./2_reco_espirit.sh
./3_reco_fastMRI_unet.sh
./4_get_measurements.sh
./5_create_us_fig.sh


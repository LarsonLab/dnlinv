#!/bin/bash

source ~/.bashrc

conda activate dnlinv

set -eo pipefail

source opts.sh

./0_reco_dnlinv.sh
./1_reco_dnlinv_fixed_noise.sh
./2_reco_dnlinv_linear_act.sh
./3_reco_dip.sh
./4_reco_dip_mc.sh
./5_plot_results.sh


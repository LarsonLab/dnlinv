#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

python plot_results.py 5-0 --recon_methods DNLINV DNLINV_fixed_noise DNLINV_linear_act DIP DIP_MC --reference_image_path data/knee-fully-sampled-reference --flip_y --figsize 11 4

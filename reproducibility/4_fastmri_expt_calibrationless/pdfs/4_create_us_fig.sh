#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

source opts.sh

python $DNLINV_PATH/create_comparison_fig.py "${USs[@]}" --recon_methods ENLIVE DNLINV FastMRI_Unet --reference_image_path data/knee-fully-sampled-reference --transpose --flip_x --figsize 8 8

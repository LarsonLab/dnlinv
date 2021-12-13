#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

python $DNLINV_PATH/create_comparison_fig.py 2-0 3-0 5-0 --recon_methods SAKE ENLIVE DIP DNLINV --reference_image_path data/knee-fully-sampled-reference --flip_y --figsize 8 8

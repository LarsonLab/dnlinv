#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

python $DNLINV_PATH/create_comparison_fig.py 2 3 4 5 --recon_methods ESPIRiT ENLIVE DIP DNLINV --reference_image_path data/knee-fully-sampled-reference --flip_y --flip_x --caipi_fig --figsize 8 10

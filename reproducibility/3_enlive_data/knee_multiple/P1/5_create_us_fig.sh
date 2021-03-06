#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

source opts.sh

# Get undersampling string names
US_STRING=""
for USind in "${!VALS[@]}"; do
    USval=${VALS[$USind]}
    US_STRING="$US_STRING $(GET_US $USval)"
done

python $DNLINV_PATH/create_comparison_fig.py $US_STRING --recon_methods ENLIVE DIP DNLINV --reference_image_path data/knee-fully-sampled-reference --flip_y --figsize 8 8

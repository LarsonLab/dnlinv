#!/bin/bash
set -euo pipefail

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

source opts.sh

python plot_results.py ${ITERS[@]}  --recon_methods DIP DNLINV SENSE --transpose  --figsize 8 36

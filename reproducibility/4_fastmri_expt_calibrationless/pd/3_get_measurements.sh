#!/bin/bash
set -euo pipefail
set -B

if [ ! -e $TOOLBOX_PATH/bart ] ; then
	echo "\$TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi
export PATH=$TOOLBOX_PATH:$PATH

if [ ! -e $DNLINV_PATH/measure_error.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

source opts.sh

if [ ! -e data/knee-fully-sampled-reference.cfl ] ; then
    bart fft -u -i $(bart bitmask 1 2) data/slice-full data/tmp_knee_full_ifft
    bart rss $(bart bitmask 3) data/tmp_knee_full_ifft data/tmp_knee-fully-sampled-reference
    bart squeeze data/tmp_knee-fully-sampled-reference data/knee-fully-sampled-reference
    rm data/tmp_knee*
fi

for US in "${USs[@]}";
do

    echo "DNLINV ${US}"
    python $DNLINV_PATH/measure_error.py data/knee-fully-sampled-reference reco_DNLINV/dnlinv_${US}/rsos_multicoil_estimate

    echo "FastMRI_Unet ${US}"
    python $DNLINV_PATH/measure_error.py data/knee-fully-sampled-reference reco_FastMRI_Unet/r_mm_${US}

    echo "ENLIVE ${US}"
    python $DNLINV_PATH/measure_error.py data/knee-fully-sampled-reference reco_ENLIVE/r_mm_${US}

done

    

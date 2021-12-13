#!/bin/bash
set -euo pipefail
set -B

if [ ! -e $TOOLBOX_PATH/bart ] ; then
	echo "\$TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi
export PATH=$TOOLBOX_PATH:$PATH

if [ ! -e $DNLINV_PATH/reconstruct.py ] ; then
	echo "\$DNLINV_PATH is not set correctly!" >&2
	exit 1
fi

source opts.sh
out=reco_DNLINV
mkdir -p $out

for US in "${USs[@]}"
do
    if [ ! -e data/pat_$US.cfl ]; then
	python $DNLINV_PATH/fastmri_gen_pattern.py $US data/pat_$US --center_fraction 0.01
    fi

    # apply undersampling
    #bart fmac data/slice-full data/pat_$US ${DATA}_$US

    python $DNLINV_PATH/reconstruct.py ${DATA}_${US} $out/dnlinv_${US} $DNLINV_OPTS --mask data/pat_${US}
done



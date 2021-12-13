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
export PATH=$DNLINV_PATH:$PATH

source opts.sh
out=reco_DNLINV
mkdir -p $out

for US in "${USs[@]}"
do
    if false; then
	./gen_pattern.py $ACS $US
    fi

    python $DNLINV_PATH/reconstruct.py ${DATA}_${US} $out/dnlinv_${US} $DNLINV_OPTS --mask data/pat_${US}
done



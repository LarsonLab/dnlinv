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
out=reco_DIP_MC
mkdir -p $out

for USind in "${!VALS[@]}";
do
	USval=${VALS[$USind]}

	if [ $USind -eq 0 ]; then
	        TASKSET="taskset -c 0"
		export OMP_NUM_THREADS=1
	else
        	TASKSET=''
	fi

	PREP $USval

	US=$(GET_US $USval)

  python reconstruct_dip_mc.py ${DATA}_${US} $out/dip_${US} $DIP_MC_OPTS --mask data/pat_${US}
done



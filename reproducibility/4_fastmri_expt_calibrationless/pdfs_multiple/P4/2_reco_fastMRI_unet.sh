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
out=reco_FastMRI_Unet
mkdir -p $out

for US in "${USs[@]}"
do
    if [ ! -e data/pat_$US.cfl ]; then
		python $DNLINV_PATH/fastmri_gen_pattern.py $US data/pat_$US --center_fraction 0.01
	fi

	# apply undersampling
	bart fmac data/slice-full data/pat_$US ${DATA}_$US

    python $DNLINV_PATH/fastmri_run_unet.py ${DATA}_${US} data/pat_${US} $out/r_mm_${US} --model_state_dict_file "../knee_mc_leaderboard_state_dict.pt"
done



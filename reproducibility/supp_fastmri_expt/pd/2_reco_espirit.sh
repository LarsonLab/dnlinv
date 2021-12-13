#!/bin/bash
set -euo pipefail
# brace expand
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
out=reco_ESPIRiT
mkdir -p $out

for US in "${USs[@]}"
do
	if [ ! -e data/pat_$US.cfl ]; then
		python $DNLINV_PATH/fastmri_gen_pattern.py $US data/pat_$US
	fi

	# apply undersampling
	bart fmac data/slice-full data/pat_$US ${DATA}_$US

	DEBUG=4
	MAPS=2

	bart ecalib -m3 ${DATA}_${US} sens_m3_${US} > /dev/null


	bart pics -d$DEBUG -S -l1 -r0.008 ${DATA}_${US} sens_m3_${US} $out/r_mm_${US} >$out/log_r_mm_${US}
	bart rss $(bart bitmask 4 10)  $out/r_mm_{,abs_}${US}

	rm sens_m3_${US}.{cfl,hdr}
done

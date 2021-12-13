#!/bin/bash

conda activate dnlinv

set -eo pipefail

for COUNT in {1..20}; do
	echo Processing P$COUNT
	cd P$COUNT
	./0_reco_dip.sh
	./1_reco_dnlinv.sh
	./2_reco_enlive.sh
	./3_reco_sake.sh
	./4_get_measurements.sh
	./5_create_us_fig.sh
	cd ..
done



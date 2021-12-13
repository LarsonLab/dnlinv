#!/bin/bash

SLURM_OPTS="--gres=gpu:teslav100 --mem=64g --partition=dgx --time=12:00:00"

for COUNT in {1..10}; do
	echo Processing P$COUNT
	cd P$COUNT
    sbatch $SLURM_OPTS slurm_doit.sh
	cd ..
done




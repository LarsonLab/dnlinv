#!/bin/bash

source opts.sh

#SENSE RECON
python reconstruct_phantom_SENSE.py phantom reco_SENSE/base_recon $SENSE_OPTS --n_iter 100 --existing_mps phantom

for ITER in ${ITERS[@]}; do
  if [ ! -d reco_SENSE/iter_$ITER ]; then
      mkdir reco_SENSE/iter_$ITER
  fi
  cp reco_SENSE/base_recon/* reco_SENSE/iter_$ITER
done
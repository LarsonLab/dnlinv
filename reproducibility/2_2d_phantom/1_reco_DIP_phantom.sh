#!/bin/bash

source opts.sh

for ITER in ${ITERS[@]}; do
  echo Reconstructing at $ITER iterations

  #DIP RECON
  python reconstruct_phantom.py phantom reco_DIP/iter_$ITER $DIP_OPTS --n_iter $ITER --existing_mps phantom
done
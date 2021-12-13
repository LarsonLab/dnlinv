#!/bin/bash

source opts.sh

for ITER in ${ITERS[@]}; do
  echo Reconstructing at $ITER iterations

  #DNLINV RECON
  python reconstruct_phantom.py phantom reco_DNLINV/iter_$ITER $DNLINV_OPTS --n_iter $ITER --existing_mps phantom
done
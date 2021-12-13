#!/bin/bash

source ~/.bashrc

conda activate dnlinv

source opts.sh

for ITER in ${ITERS[@]}; do
  echo Reconstructing $ITER
  #DNLINV RECON
  python reconstruct_phantom.py phantom reco_DNLINV/iter_$ITER $DNLINV_OPTS --n_iter $ITER

  #DIP RECON
  python reconstruct_phantom.py phantom reco_DIP/iter_$ITER $DIP_OPTS --n_iter $ITER
done

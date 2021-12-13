import os
import os.path
import sys
if 'FASTMRI_PATH' in os.environ:
    sys.path.append(os.environ['FASTMRI_PATH'])
else:
    raise EnvironmentError("FASTMRI_PATH is not set.")
if 'TOOLBOX_PATH' in os.environ:
    sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
    raise EnvironmentError("TOOLBOX_PATH is not set.")
import cfl
import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
import numpy as np
import numpy.random
import requests
import torch
from tqdm import tqdm


def main(hparams):
    # Based on: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/subsample.py#L72
    shape = hparams.img_shape
    num_cols = shape[-1]

    rng = np.random.RandomState(seed=hparams.seed)

    acceleration = hparams.acceleration
    if hparams.center_fraction is None:
        # if acceleration not in [4, 8]:
        #     raise ValueError(f"Unsupported acceleration factor {acceleration}. Expected 4 or 8.")
        center_fraction = 0.04 if acceleration == 8 else 0.08
    else:
        center_fraction = hparams.center_fraction


    # create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (
            num_cols - num_low_freqs
    )
    mask = rng.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True

    # reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols

    mask = mask * np.ones(shape, dtype=np.bool)  # Broadcast to full image shape
    mask = mask.T  # Transpose to convert to BART dimension ordering
    mask = mask[np.newaxis, ...]  # Add singleton dimension

    cfl.writecfl(hparams.output_path, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("acceleration", type=int, default=4, help="Accepts only integers.")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--center_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=34433, help="Seed for RNG")
    parser.add_argument("--img_shape", type=int, nargs=2, default=(320, 320))

    hparams = parser.parse_args()

    np.random.seed(seed=hparams.seed)
    torch.manual_seed(hparams.seed)

    main(hparams)




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
import requests
import torch
from fastmri.data import SliceDataset
from fastmri.models import Unet
from tqdm import tqdm

"""
NOTE: This script must be run with the fastMRI environment requirements
> conda create -n fastMRI python=3.8
> conda activate fastMRI
> conda install cudatoolkit=10.1
> pip install fastmri
"""

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}

def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()


def apply_mask(kspace, mask):
    return kspace * mask[np.newaxis, ...]  # Add singular coil dimension to mask


def run_inference(kspace, mask, challenge='unet_knee_mc', state_dict_file=None, device='cuda'):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    # download the state_dict if we don't have it
    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = UNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]
    if os.path.exists(state_dict_file) is not True:
        url_root = UNET_FOLDER
        download_model(url_root + MODEL_FNAMES[challenge], state_dict_file)

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()
    model = model.to(device)

    # Apply mask
    masked_kspace = apply_mask(kspace, mask)
    masked_kspace = T.to_tensor(masked_kspace)

    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(masked_kspace)

    # absolute value
    image = fastmri.complex_abs(image)
    image = fastmri.rss(image)

    # normalize input
    image, mean, std = T.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    output = model(image.to(device).unsqueeze(0).unsqueeze(1)).squeeze(1).cpu()

    output = (output * std + mean).cpu()

    return output


def main(input_path, mask_path, output_path, state_dict_file=None):
    input_kspace = cfl.readcfl(input_path)
    input_kspace = np.squeeze(input_kspace)  # Remove any singleton dimensions
    input_kspace = input_kspace.T  # Convert back to NumPy ordering from BART ordering
    mask = cfl.readcfl(mask_path)
    mask = np.squeeze(mask)
    mask = mask.T

    output_img = run_inference(input_kspace, mask, state_dict_file=state_dict_file)
    output_img = output_img.detach().numpy().squeeze()
    output_img = output_img.T  # Convert to BART ordering

    cfl.writecfl(output_path, output_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_kspace_path", type=str, help="Path to kspace raw data in BART CFL format")
    parser.add_argument("mask_path", type=str, help="Path to sampling mask in BART CFL format")
    parser.add_argument("output_path", type=str, help="Path to output in BART CFL format")
    parser.add_argument("--model_state_dict_file", type=str, help="Path to the model state dict file")
    parser.set_defaults(model_state_dict_file=None)
    args = parser.parse_args()

    main(args.input_kspace_path, args.mask_path, args.output_path, state_dict_file=args.model_state_dict_file)


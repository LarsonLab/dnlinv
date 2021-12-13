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
import fastmri
import fastmri.data.transforms as T
import numpy as np
import torch
import h5py


"""
NOTE: This script must be run with the fastMRI environment requirements
> conda create -n fastMRI python=3.8
> conda activate fastMRI
> conda install cudatoolkit=10.1
> pip install fastmri
"""


def main(hparams):
    with h5py.File(hparams.h5_path, 'r') as f:
        data = f['kspace']
        print(data.shape)
        # Take middle slice
        crop_img_shape = np.min(np.stack([hparams.img_shape, data.shape[-2:]], axis=0),
                                axis=0)  # This is so that you don't specify a crop larger than the image itself
        y_tensor = T.to_tensor(data[int(data.shape[0] / 2), :, :, ...])
        # y_tensor = transforms.complex_center_crop(y_tensor, hparams.img_shape).contiguous()
        x_tensor = fastmri.ifft2c(y_tensor)
        if hparams.no_crop is False:
            x_tensor = T.complex_center_crop(x_tensor, crop_img_shape).contiguous()
        y_tensor = fastmri.fft2c(x_tensor)
        f.close()
    out_k_space = T.tensor_to_complex_np(y_tensor)
    out_k_space = out_k_space.squeeze()  # Remove any lingering singleton dimensions
    out_k_space = out_k_space.T  # Transpose to change from NumPy ordering to BART ordering
    out_k_space = out_k_space[np.newaxis, ...]  # Add singleton z-axis dimension

    cfl.writecfl(hparams.output_path, out_k_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--img_shape", type=int, nargs=2, default=(320, 320))
    parser.add_argument("--no_crop", action='store_true', help='Enable this flag to disable cropping of data.')
    parser.set_defaults(no_crop=False)
    hparams = parser.parse_args()

    main(hparams)

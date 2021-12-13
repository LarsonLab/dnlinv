import os
# Fix for CTRL+C KeyboardInterrupt issue in Windows
# Ref: https://stackoverflow.com/questions/42653389/ctrl-c-causes-forrtl-error-200-rather-than-python-keyboardinterrupt-exception
# Ref: https://github.com/ContinuumIO/anaconda-issues/issues/905
# Ref: https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
from sys import platform
if platform == "win32":  # Need to be placed before all other imports
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import os.path
import sys

# 2D PHANTOM MODIFICATION
if os.environ['DNLINV_PATH'] is not None and os.path.exists(os.path.join(os.environ['DNLINV_PATH'], 'reconstruct.py')):
    sys.path.append(os.environ['DNLINV_PATH'])
else:
    raise EnvironmentError("DNLINV_PATH is not set correctly. Please double-check DNLINV_PATH.")



from argparse import ArgumentParser
from utils import setup_bart
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torchvision.utils
import fastMRI.data.transforms as transforms
from mridata_recon.recon_2d_fse import load_ismrmrd_to_np
import sigpy as sp
import sigpy.mri
import sigpy.mri.app
from tqdm import tqdm
import h5py
from generator_models import *
from forward_models import *
from matplotlib.pyplot import imsave
import mask_utils
import cfl
from generator_models import calculate_kl


def reconstruct(hparams):
    # Set seed
    if hparams.seed is None:
        hparams.seed = np.random.randint(2**16)

    if os.path.exists(hparams.output_dir) is not True:
        os.makedirs(hparams.output_dir)

    np.random.seed(seed=hparams.seed)
    torch.manual_seed(hparams.seed)

    # Read in data
    existing_mps = None  # Assign this to be none by default
    # if hparams.existing_mps:  # 2D PHANTOM MODIFICATION
    #     print("Using existing coil sensitivity map")
    #     existing_mps = cfl.readcfl(hparams.existing_mps)
    #     existing_mps = np.transpose(existing_mps)  # Need to transpose to have sigpy definition of [coils, kz, ky, kx] from bart definition of [kx, ky, kz, coils]
    #     existing_mps = np.transpose(existing_mps, [-1, 0, 1, 2])  # Move SENSE maps dimension to first dimension to achieve dimensions of [SENSE, coil_ch, kz, ky, kx]
    #     existing_mps = transforms.to_tensor(existing_mps).to('cuda')
    #     # raise NotImplementedError("Reading in coil sensitivity maps not yet implemented!")

    if hparams.h5_path == 'phantom': # 2D PHANTOM MODIFICATION
        num_channels = 8
        img_shape = tuple(hparams.img_shape)
        coils = transforms.to_tensor(sp.mri.birdcage_maps((num_channels,) + img_shape, r=0.8, nzz=1))
        if hparams.existing_mps:
            print("Using existing coil sensitivity map")
            existing_mps = coils.unsqueeze(0).to('cuda')
        x = transforms.to_tensor(sp.shepp_logan(img_shape)).unsqueeze(0)
        y_tensor = transforms.fft2(transforms.complex_mul(coils, x)) \
            + hparams.noise_sigma * torch.normal(torch.zeros([num_channels,
                                                              img_shape[0],
                                                              img_shape[1],
                                                              2]))
        y = sp.from_pytorch(y_tensor, iscomplex=True)
        phantom_truth = x.squeeze()
    else:
        if hparams.h5_type == 'fastMRI':
            with h5py.File(hparams.h5_path, 'r') as f:
                data = f['kspace']
                print(data.shape)
                # Take middle slice
                crop_img_shape = np.min(np.stack([hparams.img_shape, data.shape[-2:]], axis=0), axis=0)  # This is so that you don't specify a crop larger than the image itself
                y_tensor = transforms.to_tensor(data[int(data.shape[0]/2), :, :, ...])
                # y_tensor = transforms.complex_center_crop(y_tensor, hparams.img_shape).contiguous()
                x_tensor = transforms.ifft2(y_tensor)
                x_tensor = transforms.complex_center_crop(x_tensor, crop_img_shape).contiguous()
                y_tensor = transforms.fft2(x_tensor)
                f.close()
        elif hparams.h5_type == 'mrsrl-ismrmrd':
            kspace, header = load_ismrmrd_to_np(hparams.h5_path)
            print(kspace.shape)
            crop_img_shape = np.min(np.stack([hparams.img_shape, kspace.shape[-2:]], axis=0),
                                    axis=0)  # This is so that you don't specify a crop larger than the image itself
            # k-space data in an np.array of dimensions [phase, echo, slice, coils, kz, ky, kx]
            y_tensor = transforms.to_tensor(kspace[0, 0, int(kspace.shape[2]/2), ...]).squeeze()
            x_tensor = transforms.ifft2(y_tensor)
            x_tensor = transforms.complex_center_crop(x_tensor, crop_img_shape).contiguous()
            y_tensor = transforms.fft2(x_tensor)
        elif hparams.h5_type == 'bart':

            # import cfl
            kspace = cfl.readcfl(hparams.h5_path)
            kspace = np.transpose(kspace)  # Need to transpose to have sigpy definition of [coils, kz, ky, kx] from bart definition of [kx, ky, kz, coils]
            kspace = np.squeeze(kspace)  # squeeze singleton dimensions for convenience
            print(kspace.shape)
            if kspace.ndim == 4:
                crop_img_shape = np.min(np.stack([hparams.img_shape, kspace.shape[-2:]], axis=0),
                                        axis=0)  # This is so that you don't specify a crop larger than the image itself
                # k-space data in an np.array of dimensions [coils, kz, ky, kx]
                y_tensor = transforms.to_tensor(kspace[:, int(kspace.shape[2] / 2), ...]).squeeze()
            elif kspace.ndim == 3:
                crop_img_shape = np.min(np.stack([hparams.img_shape, kspace.shape[-2:]], axis=0),
                                        axis=0)  # This is so that you don't specify a crop larger than the image itself
                # k-space data in an np.array of dimensions [coils, ky, kx]
                y_tensor = transforms.to_tensor(kspace).squeeze()
            else:
                raise NotImplementedError(f"Unsupported number of data dimensions: {kspace.ndim}")
            x_tensor = transforms.ifft2(y_tensor)
            x_tensor = transforms.complex_center_crop(x_tensor, crop_img_shape).contiguous()
            y_tensor = transforms.fft2(x_tensor)
        else:
            raise ValueError(f"Unknown h5 file format type: {hparams.h5_type}")

    # Scale k-space data
    if hparams.disable_data_scaling is not True:
        mean_scale = torch.mean(y_tensor)
        y_tensor = y_tensor - mean_scale
        std_scale = torch.std(y_tensor)
        y_tensor = y_tensor / std_scale
    y = sp.from_pytorch(y_tensor, iscomplex=True)

    print(y.shape)
    num_channels, img_shape = y.shape[0], y.shape[1:]

    if hparams.mask:
        mask = cfl.readcfl(hparams.mask)
        mask = np.transpose(mask)  # Need to transpose to have sigpy definition of [coils, kz, ky, kx] from bart definition of [kx, ky, kz, coils]
        mask = mask.squeeze()
        mask = np.broadcast_to(mask[np.newaxis, ...], [num_channels, img_shape[0], img_shape[1]])
        mask = mask.astype(np.bool)
        print(mask.shape)
    else:
        # Generate mask
        mask = mask_utils.create_mask(hparams, img_shape, num_channels)

    mps = sp.from_pytorch(existing_mps.squeeze(), iscomplex=True)
    device = sp.Device(0)
    recon = sp.mri.app.SenseRecon(y.astype(np.complex128), mps.astype(np.complex128), lamda=hparams.weight_decay, weights=mask, device=device, max_iter=hparams.n_iter)

    x_mean = sp.to_device(recon.run(), device=-1)
    mps = sp.to_device(mps, device=-1)

    coil_imgs = transforms.complex_mul(transforms.to_tensor(mps), transforms.to_tensor(x_mean))
    sos_multicoil_estimate = transforms.root_sum_of_squares_complex(coil_imgs)

    cfl.writecfl(os.path.join(hparams.output_dir, 'mps'),
                 mps.transpose())
    cfl.writecfl(os.path.join(hparams.output_dir, 'x_est'), x_mean.transpose())
    cfl.writecfl(os.path.join(hparams.output_dir, 'rsos_multicoil_estimate'), sp.from_pytorch(
            sos_multicoil_estimate, iscomplex=False
        ).transpose()
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('h5_path', type=str, help="Path to the h5 file")
    parser.add_argument('output_dir', type=str, help="Output directory for the files")
    parser.add_argument("--noise_sigma", type=float, default=1.0, help="Noise standard deviation initialization")
    parser.add_argument("--mask", type=str,
                        help="Path to existing mask to use. Uses BART CFL format")
    parser.add_argument("--n_iter", type=int, default=200,
                        help="number of iterations of optimization")
    parser.add_argument("--img_shape", type=int, nargs=2, default=[256, 256],
                        help="image shape")
    parser.add_argument("--undersampling_type", type=str, default="poisson",
                        help="Types: poisson, parallel_imaging_y, parallel_imaging_x, parallel_imaging_x_y")
    parser.add_argument("--undersampling_factor", type=float, default=3)
    parser.add_argument("--calib_region_shape", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--h5_type", type=str, default='fastMRI',
                        help="Data structure and format of the H5 file to be read.\n"
                             "Options are: 'fastMRI', 'mrsrl-ismrmrd', 'bart'")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=1e-12,
                        help="Weight decay for optimization")
    parser.add_argument("--existing_mps", type=str,
                        help="Path to existing coil sensitivity maps to use. Uses BART CFL format and "
                             "dimensions definitions.")
    parser.add_argument("--disable_data_scaling", action='store_true')

    parser.set_defaults(disable_data_scaling=False, debug=False)

    hparams = parser.parse_args()

    if hparams.mask is not None:
        if os.path.exists(hparams.mask):
            raise ValueError(f"Sampling mask cannot be accessed. Path provided: {hparams.mask}")

    setup_bart()

    reconstruct(hparams)




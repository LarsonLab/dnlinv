import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import os.path
if 'TOOLBOX_PATH' in os.environ:
    sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
    raise EnvironmentError(f"BART TOOLBOX_PATH not set. Please set TOOLBOX_PATH environment variable")
if os.environ['DNLINV_PATH'] is not None and os.path.exists(os.path.join(os.environ['DNLINV_PATH'], 'reconstruct.py')):
    sys.path.append(os.environ['DNLINV_PATH'])
else:
    raise EnvironmentError("DNLINV_PATH is not set correctly. Please double-check DNLINV_PATH.")
import cfl
import sigpy as sp
import sigpy.mri
import fastMRI.data.transforms as transforms
import skimage.metrics as metrics
import argparse
import mask_utils


parser = argparse.ArgumentParser()
parser.add_argument("iters", type=str, nargs='+',
                    help="The string identifies the different iterations to open. Must be separated by spaces. "
                         "ex. 100 200 300")
parser.add_argument("--recon_methods", type=str, nargs='+',
                    help="The different reconstruction methods to compare. ex. DNLINV ENLIVE SAKE")
parser.add_argument("--transpose", action='store_true', help="Enable to transpose images")
parser.add_argument("--flip_x", action='store_true', help="Enable to flix x axis")
parser.add_argument("--flip_y", action='store_true', help="Enable to flix y axis")
parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6),
                    help="Figure size for the plot. ex. 12 6")
args = parser.parse_args()

iters_string = args.iters
recon_methods = args.recon_methods


def transpose_flip_img(in_img, in_args):
    if in_args.transpose:
        in_img = in_img.T
    if in_args.flip_x:
        in_img = np.flip(in_img, axis=0)
    if in_args.flip_y:
        in_img = np.flip(in_img, axis=1)
    return in_img

# Load reference image
num_channels = 8
img_shape = (256, 256)
coils = transforms.to_tensor(sp.mri.birdcage_maps((num_channels,) + img_shape, r=0.8, nzz=1))
x = transforms.to_tensor(sp.shepp_logan(img_shape)).unsqueeze(0)
x_coil_imgs = transforms.complex_mul(coils, x)
x_rsos = transforms.root_sum_of_squares_complex(x_coil_imgs)
ref_img = sp.from_pytorch(x_rsos, iscomplex=False)
height, width = ref_img.shape

# Load reference mask
class MaskParams:
    pass
hparams = MaskParams()
hparams.undersampling_type = 'parallel_imaging_y'
hparams.undersampling_factor = 8
hparams.calib_region_shape = (1, 1)
mask = mask_utils.create_mask(hparams, img_shape, num_channels)
mask = mask[0]  # Select just one channel

# Build figure
fig, axes = plt.subplots(nrows=2 * len(iters_string), ncols=len(recon_methods) + 2,
                         figsize=args.figsize, dpi=150)

# Make axis 2d if not
if len(axes.shape) == 1:
    axes = axes[np.newaxis, ...]

for idx_iter, iter_val in enumerate(iters_string):
    ax = axes[2 * idx_iter, 0]
    ax.imshow(mask, cmap='gray', vmax=1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for idx_method, method in enumerate(recon_methods):
        if 'DNLINV' in method:
            input_img_path = os.path.join(f'reco_{method}', f'iter_{iter_val}', 'rsos_multicoil_estimate')
        elif 'DIP' in method:
            input_img_path = os.path.join(f'reco_{method}', f'iter_{iter_val}', 'rsos_multicoil_estimate')
        else:
            input_img_path = os.path.join(f'reco_{method}', f'iter_{iter_val}', 'rsos_multicoil_estimate')

        img = cfl.readcfl(input_img_path)
        img = np.abs(img.squeeze())
        img = transpose_flip_img(img, args)

        mse = metrics.mean_squared_error(ref_img, img)
        nrmse = metrics.normalized_root_mse(ref_img, img)
        psnr = metrics.peak_signal_noise_ratio(ref_img, img, data_range=ref_img.max())
        ssim = metrics.structural_similarity(ref_img, img, multichannel=False, data_range=ref_img.max())

        results = {'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr, 'SSIM': ssim}
        # if args.plot_error_image:
        #     sp.plot.ImagePlot(img - ref_img)
        err_img = np.abs(img - ref_img)

        ax = axes[2 * idx_iter, idx_method + 1]
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=ref_img.max() / 2)
        ax.text(0, 1.07 * height, f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}', fontsize='xx-small', color='k')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes[2 * idx_iter + 1, idx_method + 1]
        ax.imshow(3 * err_img, cmap='gray', vmax=ref_img.max() / 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# Add reference
ax = axes[0, -1]
ax.imshow(ref_img, cmap='gray', vmin=0.0, vmax=ref_img.max() / 2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('Reference')

# Hide other axes
for ax in axes[1:, -1]:
    ax.set_axis_off()

for ax in axes[1::2, 0]:
    ax.set_axis_off()

# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
# Set titles
axes[0, 0].set_title('Sampling mask', {'fontsize': 'small'})

col_titles = recon_methods
row_titles = iters_string
for ax, col in zip(axes[0, 1:], col_titles):
    ax.set_title(col)

# for ax, col in zip(axes[1::2, 1:], col_titles):
#     ax.set_title(f'3 x |{col} - Ref|',
#                  {'fontsize': 'medium'}
#                  )



for ax, row in zip(axes[0::2, 0], row_titles):
    row = row.replace('-', '.')
    ax.text(-0.2 * width, height / 2, f'Iter = {row}', rotation='vertical', fontsize='large', verticalalignment='center')
    # ax.set_ylabel(row)

fig.tight_layout()
plt.savefig('FIGURE_comparisons.png', dpi=300)
#plt.show()

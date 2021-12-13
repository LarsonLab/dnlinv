import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import os.path
if 'TOOLBOX_PATH' in os.environ:
    sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
    raise EnvironmentError(f"BART TOOLBOX_PATH not set. Please set TOOLBOX_PATH environment variable")
import cfl
import skimage.metrics as metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("undersampling_factors_string", type=str, nargs='+',
                    help="The string identifies for the different undersampling factors. Must be separated by spaces. "
                         "ex. 2-0 3-0 5-0")
parser.add_argument("--recon_methods", type=str, nargs='+',
                    help="The different reconstruction methods to compare. ex. DNLINV ENLIVE SAKE")
parser.add_argument("--reference_image_path", type=str,
                    help="Path to the reference image")
parser.add_argument("--transpose", action='store_true', help="Enable to transpose images")
parser.add_argument("--flip_x", action='store_true', help="Enable to flix x axis")
parser.add_argument("--flip_y", action='store_true', help="Enable to flix y axis")
parser.add_argument("--caipi_fig", action='store_true', help="Enable when making the figure for CAIPI to properly calculate the acceleration factor")
parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6),
                    help="Figure size for the plot. ex. 12 6")
args = parser.parse_args()

undersampling_factors_string = args.undersampling_factors_string
recon_methods = args.recon_methods
ref_img_path = args.reference_image_path

method_filenames_key = {
    'DNLINV': 'rsos_multicoil_estimate',
    'DIP': 'rsos_multicoil_estimate',
    'ENLIVE': 'r_mm_',
    'SAKE': 'r_sake_',
    'ESPIRiT': 'r_mm_abs_',
    'FastMRI_Unet': 'r_mm_',
}


def transpose_flip_img(in_img, in_args):
    if in_args.transpose:
        in_img = in_img.T
    if in_args.flip_x:
        in_img = np.flip(in_img, axis=0)
    if in_args.flip_y:
        in_img = np.flip(in_img, axis=1)
    return in_img

# Load reference image
ref_img = cfl.readcfl(ref_img_path)
ref_img = np.abs(ref_img.squeeze())
ref_img = transpose_flip_img(ref_img, args)
height, width = ref_img.shape

# Build figure
fig, axes = plt.subplots(nrows=2 * len(recon_methods) + 2, ncols=len(undersampling_factors_string),
                         figsize=args.figsize, dpi=150)

# Make axis 2d if not
if len(axes.shape) == 1:
    axes = axes[np.newaxis, ...]

for idx_us, us in enumerate(undersampling_factors_string):
    mask = cfl.readcfl(os.path.join('data', f'full_pat_{us}')) \
        if os.path.exists(os.path.join('data', f'full_pat_{us}.cfl')) \
        else cfl.readcfl(os.path.join('data', f'pat_{us}'))
    mask = np.abs(mask.squeeze())
    mask = transpose_flip_img(mask, args)
    mask /= mask.max()
    ax = axes[0, idx_us]
    ax.imshow(mask, cmap='gray', vmax=1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for idx_method, method in enumerate(recon_methods):
        if method == 'DNLINV':
            input_img_path = os.path.join(f'reco_{method}', f'dnlinv_{us}', method_filenames_key[method])
        elif method == 'DIP':
            input_img_path = os.path.join(f'reco_{method}', f'dip_{us}', method_filenames_key[method])
        else:
            input_img_path = os.path.join(f'reco_{method}', f'{method_filenames_key[method]}{us}')

        ndims_ref = len(ref_img.shape)
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

        ax = axes[2*idx_method + 1, idx_us]
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=ref_img.max() / 2)
        ax.text(0, 0.99 * height, f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}', fontsize='xx-small', color='w')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes[2*idx_method + 2, idx_us]
        ax.imshow(3 * err_img, cmap='gray', vmax=ref_img.max() / 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# Add reference
ax = axes[-1, 0]
ax.imshow(ref_img, cmap='gray', vmin=0.0, vmax=ref_img.max() / 2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# ax.set_title('Reference')
ax.text(-0.2 * width, height / 2, 'Reference', rotation='vertical', fontsize='medium', verticalalignment='center')

# Hide other axes
for ax in axes[-1, 1:]:
    ax.set_axis_off()

# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
# Set titles
# axes[0, 0].set_title('Sampling mask', {'fontsize': 'small'})
axes[0, 0].text(-0.2 * width, height / 2, 'Sampling mask', rotation='vertical', fontsize='medium', verticalalignment='center')

row_titles = recon_methods
col_titles = undersampling_factors_string
for ax, row in zip(axes[1::2, 0], row_titles):
    ax.text(-0.2 * width, height / 2, row, rotation='vertical', fontsize='medium', verticalalignment='center')

for ax, row in zip(axes[2::2, 0], row_titles):
    ax.text(-0.2 * width, height / 2, f'3 x |{row} - Ref|', rotation='vertical', fontsize='medium', verticalalignment='center')

for ax, col in zip(axes[0, :], col_titles):
    col = col.replace('-', '.')
    if args.caipi_fig:
        col = int(col) ** 2
    ax.set_title(f"AF = {col}")

fig.tight_layout()
plt.savefig('FIGURE_comparisons.png', dpi=300)
plt.show()

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
import pandas as pd
import matplotlib
matplotlib.use('Agg')

undersampling_factors_string = ["2", "3", "4"]
recon_methods = "ENLIVE DNLINV FastMRI_Unet".split()

patient_directories = [f'P{idx + 1}' for idx in range(10)]

method_filenames_key = {
    'DNLINV': 'rsos_multicoil_estimate',
    'DIP': 'rsos_multicoil_estimate',
    'ENLIVE': 'r_mm_',
    'SAKE': 'r_sake_',
    'ESPIRiT': 'r_mm_abs_',
    'FastMRI_Unet': 'r_mm_',
}

psnr_results = np.empty([len(undersampling_factors_string), len(recon_methods), len(patient_directories)])
ssim_results = np.empty([len(undersampling_factors_string), len(recon_methods), len(patient_directories)])
for idx_p, pdir in enumerate(patient_directories):
    # Load reference image
    ref_img_path = os.path.join(pdir, 'data', 'knee-fully-sampled-reference')
    ref_img = cfl.readcfl(ref_img_path)
    ref_img = np.abs(ref_img.squeeze())
    # ref_img = transpose_flip_img(ref_img, args)
    height, width = ref_img.shape

    for idx_us, us in enumerate(undersampling_factors_string):
        mask = cfl.readcfl(os.path.join(pdir, 'data', f'full_pat_{us}')) \
            if os.path.exists(os.path.join(pdir, 'data', f'full_pat_{us}.cfl')) \
            else cfl.readcfl(os.path.join(pdir, 'data', f'pat_{us}'))
        mask = np.abs(mask.squeeze())
        # mask = transpose_flip_img(mask, args)
        mask /= mask.max()

        for idx_method, method in enumerate(recon_methods):
            if method == 'DNLINV':
                input_img_path = os.path.join(pdir, f'reco_{method}', f'dnlinv_{us}', method_filenames_key[method])
            elif method == 'DIP':
                input_img_path = os.path.join(pdir, f'reco_{method}', f'dip_{us}', method_filenames_key[method])
            else:
                input_img_path = os.path.join(pdir, f'reco_{method}', f'{method_filenames_key[method]}{us}')

            ndims_ref = len(ref_img.shape)
            img = cfl.readcfl(input_img_path)
            img = np.abs(img.squeeze())
            # img = transpose_flip_img(img, args)

            mse = metrics.mean_squared_error(ref_img, img)
            nrmse = metrics.normalized_root_mse(ref_img, img)
            psnr = metrics.peak_signal_noise_ratio(ref_img, img, data_range=ref_img.max())
            ssim = metrics.structural_similarity(ref_img, img, multichannel=False, data_range=ref_img.max())

            results = {'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr, 'SSIM': ssim}

            psnr_results[idx_us, idx_method, idx_p] = psnr
            ssim_results[idx_us, idx_method, idx_p] = ssim

summary_psnr_results = np.stack([np.mean(psnr_results, axis=-1), np.std(psnr_results, axis=-1)], axis=-1)
summary_ssim_results = np.stack([np.mean(ssim_results, axis=-1), np.std(ssim_results, axis=-1)], axis=-1)

fig_size = (2, 4)
text_data = ""
for idx_us, us in enumerate(undersampling_factors_string):
    plt.figure(figsize=fig_size)
    plt.title(f'AF = {us.replace("-", ".")}')
    plt.bar(range(len(recon_methods)), summary_psnr_results[idx_us, ..., 0],
            fc="None", edgecolor="k")
    plt.xticks(range(len(recon_methods)), recon_methods)
    plt.ylabel('Peak signal-to-noise ratio (dB)')
    plt.errorbar(x=range(len(recon_methods)), y=summary_psnr_results[idx_us, ..., 0],
                 yerr=summary_psnr_results[idx_us, ..., 1], fmt='None', capsize=5, ecolor='k')
    plt.ylim([0, 40])
    plt.tight_layout()
    plt.savefig(f'summary_psnr_{us}.png', dpi=300)
    plt.close()

    plt.figure(figsize=fig_size)
    plt.title(f'AF = {us.replace("-", ".")}')
    plt.bar(range(len(recon_methods)), summary_ssim_results[idx_us, ..., 0],
            fc="None", edgecolor="k")
    plt.xticks(range(len(recon_methods)), recon_methods)
    plt.ylabel('Structural similarity index (a.u.)')
    plt.errorbar(x=range(len(recon_methods)), y=summary_ssim_results[idx_us, ..., 0],
                 yerr=summary_ssim_results[idx_us, ..., 1], fmt='None', capsize=5, ecolor='k')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'summary_ssim_{us}.png', dpi=300)
    plt.close()

    # Write data to text
    text_data = text_data + f'AF = {us.replace("-", ".")} \n'
    text_data = text_data + 'Peak signal-to-noise ratio (dB) \n'
    for idx_method, method in enumerate(recon_methods):
        text_data = text_data + f'{method} = {summary_psnr_results[idx_us, idx_method, 0]:.2f} +/- {summary_psnr_results[idx_us, idx_method, 1]:.2f} \n'

    # Write data to text
    text_data = text_data + 'Structural similarity index (a.u.) \n'
    for idx_method, method in enumerate(recon_methods):
        text_data = text_data + f'{method} = {summary_ssim_results[idx_us, idx_method, 0]:.4f} +/- {summary_ssim_results[idx_us, idx_method, 1]:.4f} \n'

print(text_data)
with open('summary_results.txt', 'w') as f:
    f.write(text_data)



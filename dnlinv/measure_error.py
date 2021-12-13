import numpy as np
import cfl
from argparse import ArgumentParser
from utils import setup_bart
import pandas as pd
import skimage.metrics as metrics
import skimage.exposure
import sigpy as sp
import sigpy.plot


def measure_error(args):
    ref_img_path = args.reference_image
    input_img_path = args.input_image
    output_csv_path = args.csv_output

    ref_img = cfl.readcfl(ref_img_path)
    ref_img = np.abs(ref_img.squeeze())
    ndims_ref = len(ref_img.shape)
    img = cfl.readcfl(input_img_path)
    img = np.abs(img.squeeze())

    if args.rescale_data:
        ref_img = ref_img / ref_img.max()
        img = img / img.max()

    if args.match_histogram:
        img = skimage.exposure.match_histograms(img, ref_img, multichannel=False)

    if ref_img.shape != img.shape:
        raise ValueError(f"Reference image and input image shapes do not match. Got "
                         f"{ref_img.shape} for the reference image and {img.shape} for the input image.")

    if ndims_ref > 2:
        raise ValueError(f"Data has more than 2 dimensions. The code only currently supports comparison on "
                         f"root-sum-of-square images. Please double-check your input data.")

    mse = metrics.mean_squared_error(ref_img, img)
    nrmse = metrics.normalized_root_mse(ref_img, img)
    psnr = metrics.peak_signal_noise_ratio(ref_img, img, data_range=ref_img.max())
    ssim = metrics.structural_similarity(ref_img, img, multichannel=False, data_range=ref_img.max())

    results = pd.DataFrame({'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr, 'SSIM': ssim}, index=[0])
    print(results)
    if args.csv_output is not None:
        print(f"Writing results to file at {output_csv_path}")
        results.to_csv(output_csv_path, index=False)

    if args.plot_error_image:
        sp.plot.ImagePlot(img - ref_img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('reference_image', type=str, help="Path to the reference image in BART CFL format.")
    parser.add_argument('input_image', type=str, help="Path to the input image in BART CFL format.")
    parser.add_argument('--csv_output', type=str, help="(optional) Path to output CSV file")
    parser.add_argument('--match_histogram', action='store_true',
                        help="Enable to match histogram to account for scaling differences.")
    parser.add_argument('--rescale_data', action='store_true',
                        help="Enable to scale the images by the max value.")
    parser.add_argument('--plot_error_image', action='store_true',
                        help="Enable to plot the difference image.")

    args = parser.parse_args()

    setup_bart()

    measure_error(args)

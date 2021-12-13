import numpy as np
import sigpy as sp
import sigpy.mri


def create_mask(hparams, img_shape, num_channels):
    """
    Creates a mask given the parameters
    :param hparams: argparse arguments
    :param img_shape: list with the image shape
    :param num_channels: number of coil channels
    :return: mask: a np.bool object with shape [num_channels] + img_shape
    """
    if hparams.undersampling_type == 'poisson':
        mask = sp.mri.poisson(img_shape=img_shape, accel=hparams.undersampling_factor,
                              calib=hparams.calib_region_shape)
        mask = np.broadcast_to(mask[np.newaxis, ...], [num_channels, img_shape[0], img_shape[1]])
        mask = mask.astype(np.bool)
    elif hparams.undersampling_type == 'parallel_imaging_y':
        accel_factor = int(hparams.undersampling_factor)
        calib_shape_y, calib_shape_x = hparams.calib_region_shape
        mask = np.zeros([num_channels, img_shape[0], img_shape[1]], dtype=np.bool)
        mask[:, ::accel_factor, :] = 1
        mask[:,
        int(img_shape[0] / 2 - calib_shape_y / 2):int(img_shape[0] / 2 + calib_shape_y / 2),
        int(img_shape[1] / 2 - calib_shape_x / 2):int(img_shape[1] / 2 + calib_shape_x / 2)
        ] = 1
    elif hparams.undersampling_type == 'parallel_imaging_x':
        accel_factor = int(hparams.undersampling_factor)
        calib_shape_y, calib_shape_x = hparams.calib_region_shape
        mask = np.zeros([num_channels, img_shape[0], img_shape[1]], dtype=np.bool)
        mask[:, :, ::accel_factor] = 1
        mask[:,
        int(img_shape[0] / 2 - calib_shape_y / 2):int(img_shape[0] / 2 + calib_shape_y / 2),
        int(img_shape[1] / 2 - calib_shape_x / 2):int(img_shape[1] / 2 + calib_shape_x / 2)
        ] = 1
    elif hparams.undersampling_type == 'parallel_imaging_x_y':
        accel_factor = int(hparams.undersampling_factor)
        calib_shape_y, calib_shape_x = hparams.calib_region_shape
        mask = np.zeros([num_channels, img_shape[0], img_shape[1]], dtype=np.bool)
        mask_x = mask.copy()
        mask_y = mask.copy()
        mask_x[:, :, ::accel_factor] = 1
        mask_y[:, ::accel_factor, :] = 1
        mask = mask_x * mask_y
        mask[:,
        int(img_shape[0] / 2 - calib_shape_y / 2):int(img_shape[0] / 2 + calib_shape_y / 2),
        int(img_shape[1] / 2 - calib_shape_x / 2):int(img_shape[1] / 2 + calib_shape_x / 2)
        ] = 1
    else:
        raise NameError(f"Unknown undersampling type: {hparams.undersampling_type}")

    return mask
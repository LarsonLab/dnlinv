import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import fastMRI.data.transforms as transforms


class ForwardModel(nn.Module):
    def __init__(self, noise_sigma, num_channels, img_shape, mask, mps, device='cpu'):
        super(ForwardModel, self).__init__()

        self.num_channels = num_channels
        self.C = transforms.to_tensor(mps).unsqueeze(0).to(device)
        self.img_shape = img_shape
        self.device = device
        self.mask = mask
        self.rng = torch.zeros([1,
                                self.num_channels,
                                self.img_shape[0],
                                self.img_shape[1], 2]).to(device)

    def forward(self, image, noise_sigma):
        y = transforms.fft2(transforms.complex_mul(self.C.unsqueeze(0), image.unsqueeze(1))) \
            + noise_sigma * self.rng.normal_()
        return y[self.mask.expand_as(y)].reshape(image.shape[0], -1)


class ForwardModelCoilEstimated(nn.Module):
    def __init__(self, noise_sigma, num_channels, img_shape, mask, maximum_likelihood=False, device='cpu',
                 n_mps=1):
        super(ForwardModelCoilEstimated, self).__init__()

        self.num_channels = num_channels
        self.img_shape = img_shape
        self.device = device
        self.mask = mask.to(device)
        self.n_mps = n_mps
        self.rng = torch.zeros([self.n_mps,
                                self.num_channels,
                                self.img_shape[0],
                                self.img_shape[1], 2]).to(device)
        self.rng.requires_grad = False
        self.maximum_likelihood = maximum_likelihood


    def forward(self, image, coil_est, noise_sigma):
        if self.maximum_likelihood:
            y = transforms.fft2(transforms.complex_mul(coil_est, image.unsqueeze(2)))
            y = torch.sum(y, dim=1)  # Reduce Soft SENSE dim
        else:
            y = transforms.fft2(transforms.complex_mul(coil_est, image.unsqueeze(2)))
            y = torch.sum(y, dim=1)  # Reduce Soft SENSE dim
            y = y + noise_sigma * self.rng.normal_()

        return y[self.mask.expand_as(y)].reshape(image.shape[0], self.num_channels, -1)


class ForwardModelCoilEstimatedNoiseCovariance(nn.Module):
    def __init__(self, noise_sigma, num_channels, img_shape, mask, maximum_likelihood=False, device='cpu',
                 n_mps=1):
        super(ForwardModelCoilEstimatedNoiseCovariance, self).__init__()

        self.num_channels = num_channels
        self.img_shape = img_shape
        self.device = device
        self.mask = mask.to(device)
        self.n_mps = n_mps
        self.rng = torch.zeros([self.n_mps,
                                self.num_channels,
                                self.img_shape[0],
                                self.img_shape[1], 2]).to(device)
        self.rng.requires_grad = False
        self.maximum_likelihood = maximum_likelihood

    def forward(self, image, coil_est, cholesky_noise_sigma):
        # cholesky_noise_sigma is the cholesky decomposition of the noise covariance matrix
        if self.maximum_likelihood:
            # y = transforms.fft2(coil_est * image.unsqueeze(2))
            y = transforms.fft2(transforms.complex_mul(coil_est, image.unsqueeze(2)))
            y = torch.sum(y, dim=1)  # Reduce Soft SENSE dim
        else:
            # y = transforms.fft2(coil_est * image.unsqueeze(2))
            y = transforms.fft2(transforms.complex_mul(coil_est, image.unsqueeze(2)))
            y = torch.sum(y, dim=1)  # Reduce Soft SENSE dim
            y = y + torch.matmul(cholesky_noise_sigma, self.rng.normal_().reshape(self.num_channels, -1)).reshape(self.rng.shape).unsqueeze(0)

        return y[self.mask.expand_as(y)].reshape(image.shape[0], self.num_channels, -1)  # y has shape [num_samples, num_channels, -1]


class ForwardModelCoilEstimatedNoiseCovarianceSoftSENSE(nn.Module):
    def __init__(self, noise_sigma, num_channels, img_shape, num_mps, mask, maximum_likelihood=False, device='cpu'):
        super(ForwardModelCoilEstimatedNoiseCovarianceSoftSENSE, self).__init__()

        self.num_channels = num_channels
        self.img_shape = img_shape
        self.num_mps = num_mps
        self.device = device
        self.mask = mask.to(device)
        self.rng = torch.zeros([self.num_mps,
                                self.num_channels,
                                self.img_shape[0],
                                self.img_shape[1], 2]).to(device)
        self.rng.requires_grad = False
        self.maximum_likelihood = maximum_likelihood

    def forward(self, image, coil_est, cholesky_noise_sigma):
        # cholesky_noise_sigma is the cholesky decomposition of the noise covariance matrix
        if self.maximum_likelihood:
            y = transforms.fft2(transforms.complex_mul(coil_est * image.unsqueeze(2)))
        else:
            y = transforms.fft2(transforms.complex_mul(coil_est * image.unsqueeze(2))) \
                + torch.matmul(cholesky_noise_sigma, self.rng.normal_().reshape(self.num_channels, -1)).reshape(self.rng.shape).unsqueeze(0)
        # Reduce sum soft SENSE dimension
        y = torch.sum(y, dim=1)
        return y[self.mask.expand_as(y)].reshape(image.shape[0], self.num_channels, -1)  # y has shape [num_samples, num_channels, -1]
# Soft SENSE has dimensions [samples, coil_images, coil_channels, z, y, x, complex_channels]

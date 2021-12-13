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
from tqdm import tqdm
import h5py
from generator_models import *
from forward_models import *
from matplotlib.pyplot import imsave
import mask_utils
import cfl
from generator_models import calculate_kl


# Specify generative model
class OptimizationModel(nn.Module):
    def __init__(self, y_data, forward_model, generative_model, img_shape, latent_dim, noise_sigma, device,
                 num_samples=10, max_likelihood=False, coil_init=False, existing_mps=None, l2_reg=0,
                 n_mps=1):
        # Latent variables: A, P
        # Parameters: lamda (prior precision on A), P_prior_mean, P_prior_var
        # Variational parameters: P_params
        # Observed variables: y
        super(OptimizationModel, self).__init__()
        self.device = device
        self.img_shape = img_shape + (2,)
        self.log_noise_sigma = nn.Parameter(torch.log(torch.tensor(noise_sigma)))  # Noise standard deviation
        self.y_data = y_data.to(self.device)  # Store current observed data

        # Priors for latent variable z
        self.z_prior_mean = torch.zeros([latent_dim])
        self.z_prior_var = torch.ones([latent_dim])

        # Specify generative network
        self.g = generative_model

        # Specify computational parameters
        self.latent_dim = latent_dim  # Number of latent variables
        self.num_samples = num_samples  # Number of samples to approximate expectations
        self.N = np.prod(self.img_shape)  # Number of voxels in image

        # Specify forward model
        self.f = forward_model.to(device)
        self.f.eval()
        self.f.device = device
        for param in self.f.parameters():
            param.to(device)
            param.requires_grad = False

        self.max_likelihood = max_likelihood
        self.coil_init = coil_init

        self.existing_mps = existing_mps

        self.l2_reg = l2_reg
        self.n_mps = n_mps

        self.sigma_eps = 1e-9

        self.scale_factor = nn.Parameter(torch.tensor(0.0))


    def forward(self):
        # Get image estimate
        z, x, coil_est = self.g()
        x = x.reshape(self.num_samples, 2, self.n_mps, self.img_shape[0], self.img_shape[1])
        x = x.permute(0, 2, 3, 4, 1)
        coil_est = coil_est.reshape(self.num_samples, 2, self.n_mps, self.g.num_channels // self.n_mps,
                                    self.img_shape[0], self.img_shape[1])
        coil_est = coil_est.permute(0, 2, 3, 4, 5, 1)

        # Get noise sigma from log-noise-sigma
        self.noise_sigma = torch.exp(self.log_noise_sigma)

        if self.coil_init:
            x = x.detach()
            x.fill_(1.0)
            self.noise_sigma = self.noise_sigma.detach()

        if self.image_init:
            coil_est = coil_est.detach()
            self.noise_sigma = self.noise_sigma.detach()

        if self.existing_mps is not None:
            coil_est = coil_est.detach()
            coil_est.data = self.existing_mps

        # Apply scaling factor to help with numerical optimization
        # This is because CNNs like to have mean zero and stdev of 1.0 in their outputs, so the scaling factor
        # adjusts for the magnitudes if there is any large bulk mismatch
        coil_est = coil_est * torch.exp(self.scale_factor)

        # Go to k-space
        y = self.f(x, coil_est, self.noise_sigma)

        if self.max_likelihood:
            # Calculate loss
            mse = torch.mean((self.y_data.expand_as(y) - y)**2)
            loss = mse + self.l2_reg * (torch.sum(x ** 2) + torch.sum(coil_est ** 2))
            log_likelihood = torch.tensor([0.], requires_grad=False)
            kl = torch.tensor([0.], requires_grad=False)
        else:
            # Calculate loss
            mse = torch.sum(
                torch.mean((self.y_data.expand_as(y) - y) ** 2, dim=0))

            log_likelihood = -0.5 * torch.numel(y) * (
                torch.log(torch.tensor(2 * np.pi).to(self.device)) + 2 * (self.log_noise_sigma + self.sigma_eps)) \
                             - 0.5 * (torch.exp(-2 * (self.log_noise_sigma + self.sigma_eps))) * mse

            kl = calculate_kl(self.g.z_mean, torch.sqrt(torch.abs(self.g.z_var)), 0.0, 1.0)

            reg = self.l2_reg * (torch.sum(x**2) + torch.sum(coil_est**2))

            if hasattr(self.g, 'calculate_elbo_entropy'):
                kl += self.g.calculate_elbo_entropy()

            ELBO = log_likelihood - kl
            loss = -ELBO + reg

        return loss, log_likelihood, kl, mse


# Specify generative model
class OptimizationModelWithCovariance(nn.Module):
    def __init__(self, y_data, forward_model, generative_model, img_shape, latent_dim, noise_sigma, device,
                 num_samples=10, max_likelihood=False, existing_mps=None, l2_reg=0,
                 n_mps=1):
        # Latent variables: A, P
        # Parameters: lamda (prior precision on A), P_prior_mean, P_prior_var
        # Variational parameters: P_params
        # Observed variables: y
        super(OptimizationModelWithCovariance, self).__init__()
        self.device = device
        self.img_shape = img_shape + (2,)

        self.y_data = y_data.to(self.device)  # Store current observed data

        # Priors for latent variable z
        self.z_prior_mean = torch.zeros([latent_dim])
        self.z_prior_var = torch.ones([latent_dim])

        # Specify generative network
        self.g = generative_model

        # Make a cholesky lower triangular part as the parameters
        self.L_lower = nn.Parameter(torch.zeros(self.g.num_channels // n_mps, self.g.num_channels // n_mps))
        self.L_log_diag = nn.Parameter(torch.log(torch.diag(noise_sigma * torch.eye(self.g.num_channels // n_mps))))

        # Specify computational parameters
        self.latent_dim = latent_dim  # Number of latent variables
        self.num_samples = num_samples  # Number of samples to approximate expectations
        self.N = np.prod(self.img_shape)  # Number of voxels in image

        # Specify forward model
        self.f = forward_model.to(device)
        self.f.eval()
        self.f.device = device
        for param in self.f.parameters():
            param.to(device)
            param.requires_grad = False

        self.max_likelihood = max_likelihood
        self.covar_eps = 1e-9 * torch.eye(self.g.num_channels).to(device)
        self.covar_eps.requires_grad = False

        self.existing_mps = existing_mps

        self.l2_reg = l2_reg
        self.n_mps = n_mps

        self.scale_factor = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        # Get image estimate
        z, x, coil_est = self.g()
        x = x.reshape(self.num_samples, 2, self.n_mps, self.img_shape[0], self.img_shape[1])
        x = x.permute(0, 2, 3, 4, 1)
        coil_est = coil_est.reshape(self.num_samples, 2, self.n_mps, self.g.num_channels // self.n_mps,
                                    self.img_shape[0], self.img_shape[1])
        coil_est = coil_est.permute(0, 2, 3, 4, 5, 1)

        self.L = torch.tril(self.L_lower + torch.diag(torch.exp(self.L_log_diag)))

        if self.coil_init:
            x = x.detach()
            x.fill_(1.0)
            self.L = self.L.detach()

        if self.image_init:
            coil_est = coil_est.detach()
            self.L = self.L.detach()

        if self.existing_mps is not None:
            coil_est = coil_est.detach()
            coil_est.data = self.existing_mps

        # Apply scaling factor to help with numerical optimization
        # This is because CNNs like to have mean zero and stdev of 1.0 in their outputs, so the scaling factor
        # adjusts for the magnitudes if there is any large bulk mismatch
        coil_est = coil_est * torch.exp(self.scale_factor)

        if self.max_likelihood:
            # Go to k-space
            y = self.f(x, coil_est, torch.tril(self.L))
            # Calculate loss
            mse = torch.mean((self.y_data.expand_as(y) - y) ** 2)
            loss = mse + self.l2_reg * (torch.sum(x ** 2) + torch.sum(coil_est ** 2))
            log_likelihood = torch.tensor([0.], requires_grad=False)
            kl = torch.tensor([0.], requires_grad=False)
        else:
            # Go to k-space
            y = self.f(x, coil_est, torch.tril(self.L))
            # Calculate loss
            self.noise_covar = torch.mm(self.L, self.L.t()) + self.covar_eps  # Reconstruct positive definite noise covariance
            y_diff = self.y_data.expand_as(y) - y
            # y_diff has shape [num_samples, num_channels, v]
            # noise_covar has shape [num_channels, num_channels]
            y_diff = y_diff.transpose(-1, -2)  # y_diff will now have shape [num_samples, v, num_channels]
            y_diff = y_diff.unsqueeze(-1)  # y_diff will now have shape [num_samples, v, num_channels, 1]
            inside_exp = torch.matmul(y_diff.transpose(-1, -2), torch.cholesky_inverse(self.L).unsqueeze(0).unsqueeze(0))
            inside_exp = torch.matmul(inside_exp, y_diff)

            mse = torch.sum(y_diff ** 2)

            log_likelihood = - 0.5 * torch.numel(y) * (torch.log(torch.tensor(2 * np.pi).to(self.device))) \
                             - 0.5 * torch.numel(y) / self.g.num_channels * 2 * torch.sum(self.L_log_diag) \
                             - 0.5 * torch.sum(inside_exp)

            log_noise_prior = - torch.numel(self.noise_covar)*0.5*torch.log(torch.tensor(2*np.pi).to(self.device)) \
                              - 0.5 * torch.sum(torch.diag(self.noise_covar)**2)

            kl = calculate_kl(self.g.z_mean, torch.sqrt(torch.abs(self.g.z_var)), 0.0, 1.0)   # Need to have abs on z_var to make sure that it's positive

            reg = self.l2_reg * (torch.sum(x ** 2) + torch.sum(coil_est ** 2))

            if hasattr(self.g, 'calculate_elbo_entropy'):
                kl += self.g.calculate_elbo_entropy()

            ELBO = log_likelihood + log_noise_prior - kl
            loss = -ELBO + reg

        return loss, log_likelihood, kl, mse


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
    if hparams.existing_mps:
        print("Using existing coil sensitivity map")
        existing_mps = cfl.readcfl(hparams.existing_mps)
        existing_mps = np.transpose(existing_mps)  # Need to transpose to have sigpy definition of [coils, kz, ky, kx] from bart definition of [kx, ky, kz, coils]
        existing_mps = np.transpose(existing_mps, [-1, 0, 1, 2])  # Move SENSE maps dimension to first dimension to achieve dimensions of [SENSE, coil_ch, kz, ky, kx]
        existing_mps = transforms.to_tensor(existing_mps).to('cuda')

    if hparams.h5_type == 'fastMRI':
        with h5py.File(hparams.h5_path, 'r') as f:
            data = f['kspace']
            print(data.shape)
            # Take middle slice
            crop_img_shape = np.min(np.stack([hparams.img_shape, data.shape[-2:]], axis=0), axis=0)  # This is so that you don't specify a crop larger than the image itself
            y_tensor = transforms.to_tensor(data[int(data.shape[0]/2), :, :, ...])

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

    y_data = y[mask]
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1)

    y_data = transforms.to_tensor(y_data).unsqueeze(0).reshape(1, num_channels, -1)

    device = 'cuda'

    if hparams.maximum_likelihood:
        num_samples = 1
    else:
        num_samples = hparams.num_samples
    latent_dim = hparams.latent_dim
    noise_sigma = hparams.noise_sigma

    # Set up physics/forward model
    if hparams.noise_covariance_estimate:
        f = ForwardModelCoilEstimatedNoiseCovariance(noise_sigma=noise_sigma, num_channels=num_channels, img_shape=img_shape,
                                                     mask=mask, device=device, maximum_likelihood=hparams.maximum_likelihood)
    else:
        f = ForwardModelCoilEstimated(noise_sigma=noise_sigma, num_channels=num_channels, img_shape=img_shape,
                                      mask=mask, device=device, maximum_likelihood=hparams.maximum_likelihood)

    # Set up network
    if hparams.activation == 'relu':
        activation = nn.ReLU
    else:
        raise ValueError(f"Unknown activation options: {hparams.activation}")
    if hparams.model_type == 'dcgan':
        dcgan_depth = int(hparams.network_params[0])
        g = DCGAN_G_512x512_with_coil_est(latent_dim=latent_dim, num_samples=num_samples,
                                          dcgan_depth=dcgan_depth, img_shape=img_shape,
                                          num_channels=num_channels, dropout_p=hparams.dropout_prob,
                                          momentum=hparams.batch_norm_momentum, max_likelihood=hparams.maximum_likelihood)
    elif hparams.model_type == 'unet':
        chans, num_pool_layers = hparams.network_params
        chans = int(chans)
        num_pool_layers = int(num_pool_layers)

        g = UnetModelWithCoilEst(img_shape=img_shape, latent_dim=latent_dim, in_chans=chans, out_chans=2 * hparams.n_mps,
                                 chans=chans,
                                 num_channels=num_channels * hparams.n_mps,
                                 coil_fov_oversampling=hparams.coil_fov_oversampling,
                                 interpolation_type=hparams.coil_interp_type,
                                 num_samples=num_samples,
                                 num_pool_layers=num_pool_layers, drop_prob=hparams.dropout_prob,
                                 max_likelihood=hparams.maximum_likelihood,
                                 norm_layer=nn.InstanceNorm2d,
                                 activation=activation,
                                 dip_stdev=hparams.dip_stdev,
                                 coil_num_upsample=hparams.coil_num_upsample,
                                 coil_last_conv_size=hparams.coil_last_conv_size,
                                 coil_last_conv_pad=hparams.coil_last_conv_pad)

    g = g.to(device)

    # Set up optimization model
    if hparams.noise_covariance_estimate:
        my_model = OptimizationModelWithCovariance(y_data=y_data, forward_model=f, generative_model=g, img_shape=img_shape,
                                                   latent_dim=latent_dim,
                                                   noise_sigma=noise_sigma, device=device, num_samples=num_samples,
                                                   max_likelihood=hparams.maximum_likelihood, existing_mps=existing_mps,
                                                   l2_reg=hparams.l2_reg, n_mps=hparams.n_mps)
    else:
        my_model = OptimizationModel(y_data=y_data, forward_model=f, generative_model=g, img_shape=img_shape,
                                     latent_dim=latent_dim,
                                     noise_sigma=noise_sigma, device=device, num_samples=num_samples,
                                     max_likelihood=hparams.maximum_likelihood, existing_mps=existing_mps,
                                     l2_reg=hparams.l2_reg, n_mps=hparams.n_mps)

    my_model = my_model.to(device)

    # Do optimization
    max_iter = hparams.n_iter
    optimizer_to_use = torch.optim.AdamW

    lr = hparams.lr

    if hparams.maximum_likelihood:
        optim = optimizer_to_use([{'params': my_model.g.parameters(), 'weight_decay': hparams.weight_decay},
                                  {'params': my_model.scale_factor}],
                                 lr=lr)
    else:
        if hasattr(my_model.g, 'calculate_elbo_entropy'):
            net_params_no_var = []
            net_params_var_only = []
            # Separate out variance terms
            for name, p in my_model.g.named_parameters():
                if 'rho' in name:
                    net_params_var_only.append(p)
                else:
                    net_params_no_var.append(p)
            if hparams.noise_covariance_estimate:
                optim = optimizer_to_use([{'params': net_params_no_var, 'weight_decay': hparams.weight_decay},
                                          {'params': net_params_var_only},
                                          {'params': my_model.L_log_diag},
                                          {'params': my_model.L_lower},
                                          {'params': my_model.scale_factor}],
                                         lr=lr)
            else:
                optim = optimizer_to_use([{'params': net_params_no_var, 'weight_decay': hparams.weight_decay},
                                          {'params': net_params_var_only},
                                          {'params': my_model.log_noise_sigma},
                                          {'params': my_model.scale_factor}],
                                         lr=lr)
        else:
            if hparams.noise_covariance_estimate:
                optim = optimizer_to_use([{'params': my_model.g.parameters(), 'weight_decay': hparams.weight_decay},
                                          {'params': my_model.L_log_diag},
                                          {'params': my_model.L_lower},
                                          {'params': my_model.scale_factor}],
                                         lr=lr)
            else:
                optim = optimizer_to_use([{'params': my_model.g.parameters(), 'weight_decay': hparams.weight_decay},
                                          {'params': my_model.log_noise_sigma},
                                          {'params': my_model.scale_factor}],
                                         lr=lr)
    if hparams.enable_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=hparams.cosine_scheduler_T0,
                                                                         T_mult=hparams.cosine_scheduler_Tmult)

    print("Initializing coils")
    my_model.coil_init = True
    my_model.image_init = False
    with tqdm(total=hparams.coil_init_max_iter, desc='Optimize q(z)') as pbar:
        for i in range(hparams.coil_init_max_iter):
            loss, log_likelihood, kl, mse = my_model.forward()
            optim.zero_grad()
            loss.backward(retain_graph=False)
            optim.step()

            if hparams.enable_cosine_scheduler:
                scheduler.step(i)  # For lr_scheduler.CosineAnnealingWarmRestarts

            pbar.set_postfix(iteration=i, loss=loss.item(), neg_log_likelihood=-log_likelihood.item(),
                             kl=kl.item(), mse=mse.item())
            pbar.update()

    print("Initializing image")
    my_model.coil_init = False
    my_model.image_init = True
    with tqdm(total=hparams.image_init_max_iter, desc='Optimize q(z)') as pbar:
        for i in range(hparams.image_init_max_iter):
            loss, log_likelihood, kl, mse = my_model.forward()
            optim.zero_grad()
            loss.backward(retain_graph=False)
            optim.step()

            if hparams.enable_cosine_scheduler:
                scheduler.step(i)  # For lr_scheduler.CosineAnnealingWarmRestarts

            pbar.set_postfix(iteration=i, loss=loss.item(), neg_log_likelihood=-log_likelihood.item(),
                             kl=kl.item(), mse=mse.item())
            pbar.update()

    print("Running main optimization")
    my_model.coil_init = False
    my_model.image_init = False

    if hparams.save_model_iter > 0:
        os.mkdir(os.path.join(hparams.output_dir, 'saved_models'))

    if hparams.debug:
        # Solve for A
        with tqdm(total=max_iter, desc='Optimize q(z)') as pbar:
            for i in range(max_iter):
                loss, log_likelihood, kl, mse = my_model.forward()
                optim.zero_grad()
                loss.backward(retain_graph=False)
                optim.step()

                if hparams.enable_cosine_scheduler:
                    scheduler.step(i)  # For lr_scheduler.CosineAnnealingWarmRestarts

                pbar.set_postfix(iteration=i, loss=loss.item(), neg_log_likelihood=-log_likelihood.item(),
                                 kl=kl.item(), mse=mse.item())
                pbar.update()

    else:
        try:
            # Solve for A
            with tqdm(total=max_iter, desc='Optimize q(z)') as pbar:
                for i in range(max_iter):
                    loss, log_likelihood, kl, mse = my_model.forward()
                    optim.zero_grad()
                    loss.backward(retain_graph=False)
                    optim.step()

                    if hparams.enable_cosine_scheduler:
                        scheduler.step(i)  # For lr_scheduler.CosineAnnealingWarmRestarts

                    pbar.set_postfix(iteration=i, loss=loss.item(), neg_log_likelihood=-log_likelihood.item(),
                                     kl=kl.item(), mse=mse.item())
                    pbar.update()

                    if hparams.save_model_iter > 0:
                        if i % hparams.save_model_iter == 0:
                            print(f"\nSaving model at iteration {i}\n")
                            torch.save(my_model.g.state_dict(), os.path.join(hparams.output_dir, 'saved_models', f'model_{i}.model'))

        except KeyboardInterrupt:
            print('Keyboard interrupt! Early stopping optimization.')
            pass
        finally:
            #### Freeze parameters
            for param in my_model.parameters():
                param.requires_grad = False
            for param in my_model.g.parameters():
                param.requires_grad = False

            if hparams.maximum_likelihood:
                my_model.g.eval()

            #### Freeze batch norm layers
            def set_bn_eval(module):
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()

            my_model.g.apply(set_bn_eval)

            # Do inference
            if hparams.maximum_likelihood:
                z, x, mps = my_model.g()
                x = x.reshape(num_samples, 2, hparams.n_mps, img_shape[0], img_shape[1])
                x = x.permute(0, 2, 3, 4, 1).contiguous()
                if existing_mps is not None:
                    mps = existing_mps.to('cpu').contiguous()
                else:
                    mps = mps.reshape(num_samples, 2, hparams.n_mps, num_channels, img_shape[0], img_shape[1])
                    mps = mps.permute(0, 2, 3, 4, 5, 1)
                    mps = torch.mean(mps, dim=0)
                    mps = mps.to('cpu').contiguous()
            else:
                # Generate estimate from Monte-Carlo
                n_samples = hparams.infer_num_samples
                x = []
                mps = []
                for _ in tqdm(range(n_samples), desc='Generating Monte Carlo samples'):
                    z, x_mc, coil_mc_est = my_model.g.infer()
                    x.append(x_mc.to('cpu'))
                    if existing_mps is None:
                        mps.append(coil_mc_est.to('cpu'))
                x = torch.cat(x, dim=0).reshape(n_samples, 2, hparams.n_mps, img_shape[0], img_shape[1]).permute(0, 2, 3, 4, 1).contiguous()
                np.save(os.path.join(hparams.output_dir, 'x_abs_mc_estimates.npy'), transforms.complex_abs(x).detach().numpy())
                np.save(os.path.join(hparams.output_dir, 'x_mc_estimates.npy'), sp.from_pytorch(x, iscomplex=True))

                if existing_mps is not None:
                    mps = existing_mps.to('cpu').contiguous()
                else:
                    mps = torch.cat(mps, dim=0)
                    mps = mps.reshape(n_samples, 2, hparams.n_mps, num_channels, img_shape[0], img_shape[1])
                    mps = mps.permute(0, 2, 3, 4, 5, 1)
                    mps = torch.mean(mps, dim=0).to('cpu')

            # Reapply scale factor
            mps = mps * torch.exp(my_model.scale_factor.to('cpu').detach().squeeze(0))

            coil_imgs = transforms.complex_mul(mps, torch.mean(x.unsqueeze(2), dim=0).to('cpu').expand_as(mps))

            # Re-scale data
            if hparams.disable_data_scaling is not True:
                coil_imgs = coil_imgs * std_scale
                coil_imgs = coil_imgs + mean_scale

            multicoil_estimate = torch.sum(coil_imgs, dim=0)
            sos_multicoil_estimate = transforms.root_sum_of_squares_complex(multicoil_estimate)
            x_mean = torch.mean(transforms.complex_abs(torch.sum(x, dim=1)), dim=0).to('cpu')

            # Save noise covariance matrix estimate
            if hparams.maximum_likelihood is not True:
                if hparams.noise_covariance_estimate:
                    np.savetxt(os.path.join(hparams.output_dir, "noise_covariance.txt"),
                               my_model.noise_covar.to('cpu').detach().numpy(), fmt="%.4e")
                else:
                    my_model.noise_sigma = torch.exp(my_model.log_noise_sigma)
                    print(f"Noise sigma estimate = {my_model.noise_sigma: .4e}")
                    np.savetxt(os.path.join(hparams.output_dir, "noise_sigma.txt"),
                               np.array([my_model.noise_sigma.to('cpu').detach().numpy()]), fmt="%.4e")

            if hparams.log_images:
                # Save per coil images
                for i in range(hparams.n_mps):
                    np.save(os.path.join(hparams.output_dir, f'mps_{i}.npy'), sp.from_pytorch(mps[i, ...], iscomplex=True))
                    imsave(os.path.join(hparams.output_dir, f'mps_{i}.png'),
                           np.transpose(torchvision.utils.make_grid(
                               transforms.complex_abs(mps[i, ...]).unsqueeze(1), scale_each=True, normalize=True).detach().numpy(),
                                        (1, 2, 0)),
                           origin='lower')

                    x_mean = torch.mean(transforms.complex_abs(x[:, i, ...]), dim=0).to('cpu')
                    np.save(os.path.join(hparams.output_dir, f'x_est_{i}.npy'), x_mean.detach().numpy())
                    imsave(os.path.join(hparams.output_dir, f'x_est_{i}.png'), x_mean.detach().numpy(), cmap='gray', origin='lower')

                    if hparams.maximum_likelihood is not True:
                        x_std = torch.std(transforms.complex_abs(x[:, i, ...]), dim=0).to('cpu')
                        np.save(os.path.join(hparams.output_dir, f'x_std_{i}.npy'), x_std.detach().numpy())
                        imsave(os.path.join(hparams.output_dir, f'x_std_{i}.png'), x_std.detach().numpy(), cmap='gray', origin='lower')

                    multicoil_estimate = transforms.complex_mul(mps[i, ...], torch.mean(x[:, i, ...].unsqueeze(1), dim=0).to('cpu'))
                    sos_multicoil_estimate = transforms.root_sum_of_squares_complex(multicoil_estimate)
                    np.save(os.path.join(hparams.output_dir, f'sos_multicoil_estimate_map_{i}.npy'), sp.from_pytorch(
                        sos_multicoil_estimate, iscomplex=False
                    )
                            )
                    imsave(os.path.join(hparams.output_dir, f'sos_multicoil_estimate_map_{i}.png'),
                           np.abs(sp.from_pytorch(sos_multicoil_estimate)), cmap='gray',
                           origin='lower')

                np.save(os.path.join(hparams.output_dir, f'mps.npy'), sp.from_pytorch(torch.sum(mps, dim=0), iscomplex=True))
                imsave(os.path.join(hparams.output_dir, f'mps.png'),
                       np.transpose(torchvision.utils.make_grid(
                           transforms.complex_abs(torch.sum(mps, dim=0)).unsqueeze(1), scale_each=True,
                           normalize=True).detach().numpy(),
                                    (1, 2, 0)),
                       origin='lower')

                np.save(os.path.join(hparams.output_dir, f'x_est.npy'), x_mean.detach().numpy())
                imsave(os.path.join(hparams.output_dir, f'x_est.png'), x_mean.detach().numpy(), cmap='gray',
                       origin='lower')
                if hparams.maximum_likelihood is not True:
                    x_std = torch.std(transforms.complex_abs(torch.sum(x, dim=1)), dim=0).to('cpu')
                    np.save(os.path.join(hparams.output_dir, f'x_std.npy'), x_std.detach().numpy())
                    imsave(os.path.join(hparams.output_dir, f'x_std.png'), x_std.detach().numpy(), cmap='gray',
                           origin='lower')

                np.save(os.path.join(hparams.output_dir, 'rsos_multicoil_estimate.npy'), sp.from_pytorch(
                    sos_multicoil_estimate, iscomplex=False
                )
                        )

                imsave(os.path.join(hparams.output_dir, 'rsos_multicoil_estimate.png'),
                       np.abs(sp.from_pytorch(sos_multicoil_estimate)), cmap='gray',
                       origin='lower')

            cfl.writecfl(os.path.join(hparams.output_dir, 'mps'),
                         sp.from_pytorch(torch.sum(mps, dim=0), iscomplex=True).transpose())
            cfl.writecfl(os.path.join(hparams.output_dir, 'x_est'), x_mean.detach().numpy().transpose())
            cfl.writecfl(os.path.join(hparams.output_dir, 'rsos_multicoil_estimate'), sp.from_pytorch(
                    sos_multicoil_estimate, iscomplex=False
                ).transpose()
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('h5_path', type=str, help="Path to the h5 file")
    parser.add_argument('output_dir', type=str, help="Output directory for the files")
    parser.add_argument("--mask", type=str,
                        help="Path to existing mask to use. Uses BART CFL format")
    parser.add_argument("--noise_sigma", type=float, default=1.0, help="Noise standard deviation initialization")
    parser.add_argument("--model_type", type=str, default='unet',
                        help="Model type. Options are 'dcgan', 'unet', 'linear'.")
    parser.add_argument("--num_samples", type=int, default=4, help="number of monte carlo samples")
    parser.add_argument("--infer_num_samples", type=int, default=1000, help="number of monte carlo samples on inference")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum of gradient")
    parser.add_argument("--cosine_scheduler_T0", type=int, default=100)
    parser.add_argument("--cosine_scheduler_Tmult", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_iter", type=int, default=200,
                        help="number of iterations of optimization")
    parser.add_argument("--img_shape", type=int, nargs=2, default=[256, 256],
                        help="image shape")
    parser.add_argument("--network_params", type=float, nargs='+',
                        help="List of parameters to feed to the network. "
                             "For 'dcgan', parameters would be number of the dcgan channels. Ex. --network_params 32 \n "
                             "For 'unet', parameters would be the U-net depth and number of pool layers. "
                             "Ex. --network_params 32 4")
    parser.add_argument("--dropout_prob", type=float, default=0.)
    parser.add_argument("--batch_norm_momentum", type=float, default=0.1)
    parser.add_argument("--undersampling_type", type=str, default="poisson",
                        help="Types: poisson, parallel_imaging_y, parallel_imaging_x, parallel_imaging_x_y")
    parser.add_argument("--undersampling_factor", type=float, default=3)
    parser.add_argument("--calib_region_shape", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--maximum_likelihood", action='store_true')
    parser.add_argument("--enable_cosine_scheduler", action='store_true')
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--h5_type", type=str, default='fastMRI',
                        help="Data structure and format of the H5 file to be read.\n"
                             "Options are: 'fastMRI', 'mrsrl-ismrmrd', 'bart'")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--noise_covariance_estimate", action='store_true',
                        help="Toggle true to estimate noise covariance when data is not pre-whitened")
    parser.add_argument("--n_mps", type=int, default=1,
                        help="Number of resolution levels for coil sensitivity map")
    parser.add_argument("--weight_decay", type=float, default=1e-12,
                        help="Weight decay for optimization")
    parser.add_argument("--coil_fov_oversampling", type=float, default=1.5,
                        help="Amount of oversampling of the FOV for coil sensitivity estimation FOV. "
                             "This is to minimize edge effects with the coil sensitivity estimation")
    parser.add_argument("--coil_interp_type", type=str, default='sinc',
                        help="Interpolation type to upsample the coil sensitivity maps. "
                             "'bicubic', 'sinc' (default)")
    parser.add_argument("--coil_init_max_iter", type=int, default=300,
                        help="Number of iterations for coil sensitivity estimation branch initialization")
    parser.add_argument("--image_init_max_iter", type=int, default=300,
                        help="Number of iterations for image estimation branch initialization")
    parser.add_argument("--save_model_iter", type=int, default=0,
                        help="Specifies how often to save the model. Set to 0 to disable saving.")
    parser.add_argument("--activation", type=str, default='relu',
                        help="Activation function to use throughout the network. 'relu' or 'sine'")
    parser.add_argument("--existing_mps", type=str,
                        help="Path to existing coil sensitivity maps to use. Uses BART CFL format and "
                             "dimensions definitions.")
    parser.add_argument("--dip_stdev", type=float, default=0,
                        help="Standard deviation for deep image prior latent variable space")
    parser.add_argument("--coil_num_upsample", type=int, default=1,
                        help="Number of upsampling kernels in coil output branch. Default: 1")
    parser.add_argument("--coil_last_conv_size", type=int, default=3,
                        help="Size of final convolution in coil output branch. Default: 3")
    parser.add_argument("--coil_last_conv_pad", type=int, default=1,
                        help="Size of padding in final convolution in coil output branch. Default: 1")
    parser.add_argument("--disable_data_scaling", action='store_true')
    parser.add_argument("--l2_reg", type=float, default=0,
                        help="L2 regularization on image and coil sensitivity maps as in ENLIVE")
    parser.add_argument("--log_images", action='store_true')

    parser.set_defaults(maximum_likelihood=False, noise_covariance_estimate=False, disable_data_scaling=False,
                        debug=False, log_images=False)

    hparams = parser.parse_args()

    if os.path.exists(hparams.mask):
        raise ValueError(f"Sampling mask cannot be accessed. Path provided: {hparams.mask}")

    setup_bart()

    reconstruct(hparams)




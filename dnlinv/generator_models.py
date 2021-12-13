import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import fastMRI.data.transforms as transforms
import torch


# Copied from: https://github.com/kumar-shridhar/PyTorch-BayesianCNN
def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * (torch.log(torch.tensor(sig_p)) - torch.log(sig_q)) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()  # Modified to have a bit more numerical stability
    return kl


def rfft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = transforms.ifftshift(data, dim=(-3, -2))
    data = torch.rfft(data, 2, normalized=True, onesided=False)
    data = transforms.fftshift(data, dim=(-3, -2))
    return data


def irfft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    data = transforms.ifftshift(data, dim=(-3, -2))
    data = torch.irfft(data, 2, normalized=True, onesided=False)
    data = transforms.fftshift(data, dim=(-3, -2))
    return data


def sinc_interpolation(in_data, output_size):
    num_samples, channels, y, x = in_data.shape
    in_data = in_data.reshape(num_samples, 2, channels // 2, y, x)
    in_data = in_data.permute(0, 2, 3, 4, 1)
    fft_data = transforms.fft2(in_data)
    # fft_data = rfft2(in_data)

    left_y_pad = (output_size[0] - y) // 2
    right_y_pad = (output_size[0] - y) // 2
    left_x_pad = (output_size[1] - x) // 2
    right_x_pad = (output_size[1] - x) // 2

    if (y % 2) != 0:
        left_y_pad += 1
    if (x % 2) != 0:
        left_x_pad += 1

    padded_fft_data = F.pad(fft_data, (0, 0, left_x_pad, right_x_pad, left_y_pad, right_y_pad), mode='constant', value=0)  # Weird convention for pad

    sinc_interpolated_data = transforms.ifft2(padded_fft_data)
    # sinc_interpolated_data = irfft2(padded_fft_data)

    sinc_interpolated_data = sinc_interpolated_data.permute(0, 4, 1, 2, 3)
    sinc_interpolated_data = sinc_interpolated_data.reshape(num_samples, channels, output_size[0], output_size[1])
    return sinc_interpolated_data


class DCGAN_G_512x512(nn.Module):
    def __init__(self, latent_dim, dcgan_depth, img_shape, dropout_p=0.1, momentum=0.1):
        super(DCGAN_G_512x512, self).__init__()
        self.img_shape = img_shape
        ngf = dcgan_depth
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 64, 4, 1, 0, bias=False),  # Added one more to get to 512 x 512
            nn.BatchNorm2d(ngf * 64),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 1, 0, bias=False),  # Added one more to get to 256 x 256
            nn.BatchNorm2d(ngf * 32),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.Conv2d(ngf, 2, 3, 1, 1, bias=False)
            # nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return F.interpolate(output, size=self.img_shape, mode='bilinear', align_corners=False)


class DCGAN_G_512x512_with_coil_est(nn.Module):
    def __init__(self, latent_dim, dcgan_depth, img_shape, num_channels, num_samples=1, dropout_p=0.5, momentum=0.1,
                 max_likelihood=False):
        super(DCGAN_G_512x512_with_coil_est, self).__init__()
        self.img_shape = img_shape
        self.num_channels = num_channels
        ngf = dcgan_depth
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 64, 4, 1, 0, bias=False),  # Added one more to get to 512 x 512
            nn.BatchNorm2d(ngf * 64, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),  # Added one more to get to 256 x 256
            nn.BatchNorm2d(ngf * 32, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),  # STRIDE WAS WRONG HERE
            nn.BatchNorm2d(ngf * 16, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.Conv2d(ngf, 2, 3, 1, 1, bias=False)
            # nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.coil_est = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 64, 4, 1, 0, bias=False),  # Added one more to get to 256 x 256
            nn.BatchNorm2d(ngf * 64, momentum=momentum),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(True),
            # nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),  #
            # nn.BatchNorm2d(ngf * 32),  #
            # nn.Dropout2d(p=dropout_p),  #
            # nn.ReLU(True),  #
            # nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.Conv2d(ngf * 64, 2 * num_channels, 3, 1, 1, bias=False)
            # nn.ReLU(True),
        )

        self.z_mean = nn.Parameter(torch.zeros([latent_dim]))
        self.z_var = nn.Parameter(torch.ones([latent_dim]))

        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.rng = nn.Parameter(torch.zeros([self.num_samples] + [self.latent_dim]))
        self.rng.requires_grad = False

        self.max_likelihood = max_likelihood

        if self.max_likelihood:
            self.z_var.requires_grad = False
            self.z_mean.requires_grad = False
            self.z_mean.random_()  # Initialize with a sample from a Normal distribution

    def forward(self):
        if self.max_likelihood:
            z = self.z_mean.unsqueeze(0)
        else:
            z = self.z_mean + self.z_var ** 0.5 * self.rng.normal_()
        input = z.unsqueeze(-1).unsqueeze(-1)
        output = self.main(input)
        coil_output = self.coil_est(input)
        return z, \
               F.interpolate(output, size=self.img_shape, mode='bilinear', align_corners=False), \
               F.interpolate(coil_output, size=self.img_shape, mode='bicubic', align_corners=False)

    def infer(self):  # Using only single sample for RNG
        if self.max_likelihood:
            z = self.z_mean.unsqueeze(0)
        else:
            z = self.z_mean + self.z_var ** 0.5 * self.rng[0:1].normal_()
        input = z.unsqueeze(-1).unsqueeze(-1)
        output = self.main(input)
        coil_output = self.coil_est(input)
        return z, \
               F.interpolate(output, size=self.img_shape, mode='bilinear', align_corners=False), \
               F.interpolate(coil_output, size=self.img_shape, mode='bicubic', align_corners=False)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm_layer=nn.BatchNorm2d, norm_momentum=0.1,
                 activation=nn.ReLU):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if activation == nn.ReLU or activation == nn.Identity:
            self.layers = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                norm_layer(out_chans, momentum=norm_momentum),
                activation(True),
                nn.Dropout2d(drop_prob),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                norm_layer(out_chans, momentum=norm_momentum),
                activation(True),
                nn.Dropout2d(drop_prob)
            )
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm_layer=nn.BatchNorm2d, norm_momentum=0.1,
                 activation=nn.ReLU):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        if activation == nn.ReLU or activation == nn.Identity:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
                norm_layer(out_chans, momentum=norm_momentum),
                activation(True),
                nn.Dropout2d(drop_prob),
            )
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class UnetModelWithCoilEst(nn.Module):
    """
    PyTorch implementation of a U-Net model. Adapted from fastMRI GitHub

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, img_shape, latent_dim, in_chans, out_chans, chans, num_channels, num_pool_layers, drop_prob,
                 coil_fov_oversampling=1.5, interpolation_type='sinc',
                 num_samples=1,
                 norm_layer=nn.BatchNorm2d,
                 norm_momentum=0.1,
                 activation=nn.ReLU,
                 max_likelihood=False,
                 dip_stdev=0.,
                 coil_num_upsample=1,
                 coil_last_conv_size=3,
                 coil_last_conv_pad=1):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.z_mean = nn.Parameter(torch.zeros([self.latent_dim, 1, 1]))
        self.z_var = nn.Parameter(torch.ones([self.latent_dim, 1, 1]))

        # Upsample to image shape for U-net to process
        self.latent_upconv = nn.ConvTranspose2d(self.latent_dim, self.in_chans, kernel_size=self.img_shape)

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, norm_layer=norm_layer,
                                                           norm_momentum=norm_momentum,
                                                           activation=activation)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, norm_layer=norm_layer,
                                                  norm_momentum=norm_momentum,
                                                  activation=activation)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob, norm_layer=norm_layer,
                              norm_momentum=norm_momentum,
                              activation=activation)

        coil_latent_dim = ch * 2
        ngf = chans

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch, drop_prob, norm_layer=norm_layer,
                                                          norm_momentum=norm_momentum,
                                                          activation=activation)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob, norm_layer=norm_layer,
                                       norm_momentum=norm_momentum,
                                       activation=activation)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch, drop_prob, norm_layer=norm_layer,
                                                      norm_momentum=norm_momentum,
                                                      activation=activation)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob=0., norm_layer=norm_layer,
                          norm_momentum=norm_momentum, activation=activation),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

        self.num_channels = num_channels
        self.coil_est = []
        self.coil_est += [nn.Sequential(
            nn.ConvTranspose2d(coil_latent_dim, ngf * 64, 4, 1, 0, bias=False),  # Added one more to get to 256 x 256
            norm_layer( ngf * 64, momentum=norm_momentum),
            nn.Dropout2d(p=drop_prob),
            activation(True)
        )]
        coil_kernel_ch = 64
        for _ in range(coil_num_upsample):
            self.coil_est += [
                nn.ConvTranspose2d(ngf * coil_kernel_ch,  ngf * coil_kernel_ch // 2, 2, 2, 0, bias=False),  #
            ]
            coil_kernel_ch = coil_kernel_ch // 2

        self.coil_est += [nn.Conv2d(ngf * coil_kernel_ch, 2 * self.num_channels, coil_last_conv_size, 1,
                                    coil_last_conv_pad, bias=False)]
        # Put everything together as a single sequential
        self.coil_est = nn.Sequential(*self.coil_est)

        self.coil_fov_oversampling = coil_fov_oversampling
        self.interp_type = interpolation_type
        self.num_samples = num_samples
        self.rng = nn.Parameter(torch.zeros([self.num_samples, self.latent_dim, 1, 1]), requires_grad=False)
        self.rng.requires_grad = False

        self.max_likelihood = max_likelihood

        if self.max_likelihood:
            self.z_var.requires_grad = False
            self.z_mean.requires_grad = False
            self.z_mean.random_()  # Initialize with a sample from a Normal distribution

        self.dip_stdev = dip_stdev

    def forward_model(self, z):
        input = self.latent_upconv(z)

        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        coil_latent_z = F.interpolate(output, size=[1, 1], mode='bilinear',
                                      align_corners=False)  # Compress to chans x 1 x 1
        coil_output = self.coil_est(coil_latent_z)  # Need to flatten this layer first

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        if self.interp_type == 'sinc':
            coil_output = sinc_interpolation(coil_output,
                                             output_size=(int(self.img_shape[0] * self.coil_fov_oversampling),
                                                          int(self.img_shape[1] * self.coil_fov_oversampling)))
        elif self.interp_type == 'bicubic':
            coil_output = F.interpolate(coil_output, size=(int(self.img_shape[0] * self.coil_fov_oversampling),
                                                           int(self.img_shape[1] * self.coil_fov_oversampling)),
                                        mode='bicubic', align_corners=False)
        else:
            raise Exception(f"Unknown interpolation type: {self.interp_type}")
        coil_output = transforms.center_crop(coil_output, self.img_shape)
        return z, output, coil_output

    def forward(self):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        if self.max_likelihood:
            z = self.z_mean.unsqueeze(0).expand_as(self.rng) + self.dip_stdev * self.rng.normal_()
        else:
            z = self.z_mean + torch.abs(self.z_var) ** 0.5 * self.rng.normal_()
        return self.forward_model(z)

    def infer(self):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        if self.max_likelihood:
            z = self.z_mean.unsqueeze(0)  #+ self.rng[0:1].normal_()
        else:
            z = self.z_mean + torch.abs(self.z_var) ** 0.5 * self.rng[0:1].normal_()
        return self.forward_model(z)

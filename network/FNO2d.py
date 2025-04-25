import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
import numpy as np

################################################################
# Spectral convolution
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Spectral convolution layer. Performs FFT, linear transform, and Inverse FFT.

        Parameters:
        ---
        in_channels : int,
            Number of layer input channels
        out_channels : int
            Number of layer output channels
        modes1 : int
            Number of Fourier modes to keep in the first dimension
        modes2 : int
            Number of Fourier modes to keep in the second dimension
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply in each dimension. Maximum floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        # Initialize parameter weights
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute FFT.
        # Note change in dimension from (batch_size, channels, x, y) to (batch_size, channels, x, y//2 + 1)
        # FFT output is Hermitian symmetric, so we should take the first modes1 and last modes1 to get low frequency components.
        x_ft = torch.fft.rfft2(x)
        # Initialize output FFT tensor
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat)
        # Perform complex multiplication on lower Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.complex_multiplication2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.complex_multiplication2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    @staticmethod
    def complex_multiplication2d(a, b):
        """
        Complex 2D multiplication between input tensor `a` and weight tensor `b`.

        Parameters:
        ---
        a: torch.Tensor,
            Complex input tensor of shape (batch_size, in_channels, x, y)

        b: torch.Tensor, shape (in_channel, out_channel, x, y)
            Complex weight tensor of shape (in_channels, out_channels, x, y)

        Returns:
        ---
        torch.Tensor
            Complex output tensor of shape (batch_size, out_channels, x, y)
        """
        # Sum over in_channels dimension
        return torch.einsum('bixy,ioxy->boxy', a, b)


class FourierBlock2D(nn.Module):
    """
    Single FNO block with spectral convolution, Conv2D, and activation function.

    This block performs the following operations:
    1. Applies a Fourier-based spectral convolution.
    2. Applies a 1x1 convolution in the spatial domain.
    3. Adds a residual connection between the two.
    4. Applies a GELU activation function.

    Parameters:
    ---
        width: int,
            Number of block input/output channels.
        modes1: int,
            Number of Fourier modes to use in the first dimension.
        modes2: int,
            Number of Fourier modes to use in the second dimension.
    """

    def __init__(self, width, modes1, modes2):
        super(FourierBlock2D, self).__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        device = x.device
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x = x1 + x2
        x = F.gelu(x)
        return x


class FNO2D(pl.LightningModule):
    """
    Full FNO network. It contains `num_layers` FNO blocks.

    This network performs the following operations:
    1. Lift the input channels to the desired number of channels
    2. Perform `num_layers` layers of the integral operators v' = (W + K)(v)
    3. Project the channel space to the output space

    Input:
    ---
        torch.Tensor, shape (batch_size, x, y, channels=3)
            Coefficients or initial condition and locations (a(x, y), x, y)
    Output:
    ---
        torch.Tensor, shape (batch_size, x, y, channels=1)
            Predicted solution
    """

    def __init__(self,
                 net_name="FNO2D",
                 width=32,
                 num_layers=4, 
                 modes1=8,
                 modes2=8,
                 lr=5e-4,
                 hidden_p_channels=128):
        """
        Parameters:
        ---
            width: int,
                Number of higher-dimensional channels. Default = 32.
            num_layers: int,
                Number of FNO blocks in the network. Default = 4.
            modes1: int,
                Number of Fourier modes to use in the first dimension. Default = 8.
            modes2: int,
                Number of Fourier modes to use in the second dimension. Default = 8.
            lr: float,
                Learning rate. Default = 5e-4.
            hidden_p_channels: int,
                Number of channels for the hidden layer in the projecting step. Default = 128.
        """
        super(FNO2D, self).__init__()
        self.net_name = net_name
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.hidden_p_channels = hidden_p_channels
        self.lr = lr
        self.padding = 6

        # Define affine transformation to lift 3 channels to `width` channels
        self.p = nn.Linear(3, self.width)

        # Define a list of FourierBlock2D layers
        self.fno_blocks = nn.ModuleList([
            FourierBlock2D(self.width, self.modes1, self.modes2) for _ in range(self.num_layers)
        ])

        # Define affine transformations to project the channel space to the output space
        self.q1 = nn.Linear(self.width, self.hidden_p_channels)
        self.q2 = nn.Linear(self.hidden_p_channels, 1)

        self.save_hyperparameters()

    def forward(self, x):
        # Get the grid of x
        grid = self.get_grid(x.shape, x.device)
        # Concatenate the grid to the input tensor
        x = torch.cat((x, grid), dim=-1)

        # Lift input
        x = self.p(x)
        # Permute the dimensions from (batch_size, x, y, channels) to (batch_size, channels, x, y)
        # nn.Linear operates on the last dimension
        x = x.permute(0, 3, 1, 2)

        # Perform Fourier-based spectral convolution and activation
        for layer in self.fno_blocks:
            x = layer(x)

        # Permute the dimensions from (batch_size, x, y, channels) to (batch_size, channels, x, y)
        x = x.permute(0, 2, 3, 1)

        # Project the channel space to the output space
        x = self.q1(x)
        x = F.gelu(x)
        x = self.q2(x)

        return x

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_x, size_y = shape[:-1]
        # Create grid for x and y coordinates using PyTorch
        gridx = torch.linspace(0, 1, steps=size_x).reshape(
            1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, steps=size_y).reshape(
            1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        # self.log('test_loss', loss, on_step=True, on_epoch=True)
        predictions = {'x': x, 'y': y, 'y_hat': y_hat}
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

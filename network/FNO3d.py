import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import lightning as pl
torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

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
        modes3 : int
            Number of Fourier modes to keep in the third dimension
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply in each dimension. Maximum floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    @staticmethod
    def complex_multiplication3d(a, b):
        # (batch, in_channel, x,y,z), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute FFT.
        # Note change in dimension from (batch_size, channels, x, y) to (batch_size, channels, x, y//2 + 1)
        # FFT output is Hermitian symmetric, so we should take the first modes1/modes2 and last modes1/modes2 to get low frequency components.
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.complex_multiplication3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.complex_multiplication3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.complex_multiplication3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.complex_multiplication3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    
class FourierBlock3D(nn.Module):
    """
    Single FNO block with spectral convolution, Conv3D, and activation function.

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
        modes3: int,
            Number of Fourier modes to use in the third dimension.
    """
    def __init__(self, width, modes1, modes2, modes3):
        super(FourierBlock3D, self).__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, 1)

    def forward(self, x):
        device = x.device
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x = x1 + x2
        x = F.gelu(x)
        return x


class LP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(LP, self).__init__()
        self.lp1 = nn.Linear(in_channels, mid_channels)
        self.lp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.lp1(x)
        x = F.gelu(x)
        x = self.lp2(x)
        return x

class FNO3D(pl.LightningModule):
    def __init__(self,
                 net_name='Blah',
                 in_channels=10,
                 out_channels=3,
                 modes1=8,
                 modes2=8,
                 modes3=8,
                 width=20,
                 num_layers=4,
                 lr=1e-3,
                 ):

        super(FNO3D, self).__init__()
        self.net_name = net_name

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_layers = num_layers
        self.lr = lr

        self.padding = 6

        self.p = nn.Linear(self.input_channels + 3, self.width)  # input channel is 3: (sigma(x, y, z), x, y, z)

        self.fno_blocks = nn.ModuleList([
            FourierBlock3D(self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.num_layers)
        ])

        self.q = LP(self.width, self.output_channels, self.width * 4)

        self.save_hyperparameters()

    def forward(self, x):
        # Get the grid of x
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        # Lift input
        x = self.p(x)

        # Permute the dimensions from (batch_size, x, y, z, channels) to (batch_size, channels, x, y, z)
        # nn.Linear operates on the last dimension
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # XM: delete two paddding

        for layer in self.fno_blocks:
            x = layer(x)

        x = x[..., :-self.padding]  # XM: delete two padding
        x = x.permute(0, 2, 3, 4, 1)  # XM: dimension coversion
        x = self.q(x)

        return x.float()

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_x, size_y, size_z = shape[:-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    def training_step(self, batch, batch_idx):
        sigma, y, _, _ = batch
        yhat = self(sigma)
        yhat = torch.squeeze(yhat, dim=-1)

        loss = F.mse_loss(yhat, y)
        # for j, jhat in zip([j], [jhat]):
        #     j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
        #     loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss

        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)  # rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sigma, y, _, _ = batch
        yhat = self(sigma)
        yhat = torch.squeeze(yhat, dim=-1)

        val_loss = F.mse_loss(yhat, y)
        # for j, jhat in zip([j], [jhat]):
        #     j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
        #     val_loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)  # rank_zero_only=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        sigma, y, _, _ = batch

        yhat = self(sigma)
        yhat = torch.squeeze(yhat, dim=-1)

        test_loss = F.mse_loss(yhat, y)
        # component_loss = []
        # for j, jhat in zip([j], [jhat]):
        #     j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
        #     component_loss.append(j_loss)
        #     test_loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)  # rank_zero_only=True)

        test_metrics = {
            'loss': test_loss,
            # 'component_loss': component_loss
        }

        return test_metrics

    def predict_step(self, batch, batch_idx):
        sigma, y, means, stds = batch
        yhat = self(sigma)
        print('Normal prediction range: ', yhat.min(), yhat.max())
        print('Normal truth range: ', y.min(), y.max())

        yhat = torch.squeeze(yhat, dim=-1)
        yhat = self.z_score_back_transform(yhat, means, stds)
        y = self.z_score_back_transform(y, means, stds)
        print('Back transform prediction range: ', yhat.min(), yhat.max())
        print('Back transform truth range: ', y.min(), y.max())
        # binary_predictions = (jhat <= 0).float()
        # j = (j <= 0).float()
        predictions = {
            'j': y,
            'jhat': yhat  # binary_predictions,
        }
        return predictions

    def z_score_back_transform(self, normalized_data, means, stds):
        """
        Back-transform Z-score normalized data to its original scale.
        Args:
            normalized_data (torch.Tensor): Normalized data of shape [batch_size, height, width, seq_len].
            means (np.ndarray): Means of shape [height, width, 1].
            stds (np.ndarray): Standard deviations of shape [height, width, 1].
        Returns:
            original_data (torch.Tensor): Back-transformed data of shape [batch_size, height, width, seq_len].
        """
        # Convert means and stds to PyTorch tensors
        # means = torch.tensor(means, dtype=torch.float32)  # Shape: [height, width, 1]
        # stds = torch.tensor(stds, dtype=torch.float32)    # Shape: [height, width, 1]

        # Reshape means and stds to match the batch and channel dimensions
        means = means.unsqueeze(0)  # Shape: [1, height, width, 1]
        stds = stds.unsqueeze(0)  # Shape: [1, height, width, 1]

        # Back-transform
        original_data = (normalized_data * stds) + means
        return original_data

    def back_transform_11(self, normalized_data, min_val, max_val):
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        original = (normalized_data + 1) * range_val / 2 + min_val
        return original

    def back_transform_01(self, normalized_data, min_val, max_val):
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        original = normalized_data * range_val + min_val
        return original

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor
                "interval": "epoch",  # Check every epoch
                "frequency": 5
            }
        }
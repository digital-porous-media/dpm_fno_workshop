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
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes3 = modes3  # Number of Fourier modes to multiply, at most floor(N/2) + 1

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
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)


        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
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

class GetFNO3DModel(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, width):
        super(GetFNO3DModel, self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6

        # self.encoder = OneHotEncoder(sparse_output=False).fit_transform()
        # self.decoder = OneHotEncoder(sparse_output=False).inverse_transform()
        self.p = nn.Linear(self.input_channels + 3, self.width)  # input channel is 3: (sigma(x, y, z), x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = LP(self.width, self.output_channels, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # f = x.clone()
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # XM: delete two paddding

        x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        # x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        # x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]  # XM: delete two padding
        x = x.permute(0, 2, 3, 4, 1)  # XM: dimension coversion
        x = self.q(x)
        #x = x.softmax(dim=-1)

        # TODO: Add solid phase masking
        # x[f == 2] = 2
        return x.float()

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[:-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class FNO3D(pl.LightningModule):
    def __init__(self,
                 net_name='Blah',
                 model=GetFNO3DModel,
                 in_channels=10,
                 out_channels=3,
                 modes1=8,
                 modes2=8,
                 modes3=8,
                 width=20,
                 lr=1e-3,
                 beta_1=1,
                 beta_2=0,
                 ):

        super(FNO3D, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.PE_lr = lr / 10
        self.beta_1, self.beta_2 = beta_1, beta_2

        self.model = GetFNO3DModel(in_channels=in_channels,
                                   out_channels=out_channels,
                                   modes1=modes1,
                                   modes2=modes2,
                                   modes3=modes3,
                                   width=width)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sigma, j, _, _ = batch
        jhat = self(sigma)
        jhat = torch.squeeze(jhat, dim=-1)

        loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss

        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)  # rank_zero_only=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sigma, j, _, _ = batch
        jhat = self(sigma)
        jhat = torch.squeeze(jhat, dim=-1)

        val_loss = 0
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            val_loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)  # rank_zero_only=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        sigma, j, _, _ = batch

        jhat = self(sigma)
        jhat = torch.squeeze(jhat, dim=-1)

        test_loss = 0
        component_loss = []
        for j, jhat in zip([j], [jhat]):
            j_loss = F.mse_loss(j.view((1, -1)), jhat.view((1, -1)))
            component_loss.append(j_loss)
            test_loss += self.beta_1 * j_loss  # + self.beta_2 * div_loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)  # rank_zero_only=True)

        test_metrics = {
            'loss': test_loss,
            'component_loss': component_loss
        }

        return test_metrics

    def predict_step(self, batch, batch_idx):
        sigma, j, means, stds = batch
        jhat = self(sigma)
        print('Normal prediction range: ', jhat.min(), jhat.max())
        print('Normal truth range: ', j.min(), j.max())

        jhat = torch.squeeze(jhat, dim=-1)
        jhat = self.z_score_back_transform(jhat, means, stds)
        j = self.z_score_back_transform(j, means, stds)
        print('Back transform prediction range: ', jhat.min(), jhat.max())
        print('Back transform truth range: ', j.min(), j.max())
        # binary_predictions = (jhat <= 0).float()
        # j = (j <= 0).float()
        predictions = {
            'j': j,
            'jhat': jhat  # binary_predictions,
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
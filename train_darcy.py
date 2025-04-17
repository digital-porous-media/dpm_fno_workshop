import torch
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Dataset

from FNO2d import FNO2d

import matplotlib.pyplot as plt
import random


class FractureGradientDataset(Dataset):
    def __init__(self, num_samples, size=(64, 64), fracture_width_range=(2, 10)):
        self.num_samples = num_samples
        self.size = size
        self.fracture_width_range = fracture_width_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fracture = self.generate_fracture(self.size, self.fracture_width_range)
        gradient = self.generate_linear_gradient(fracture)

        # Add channel dimension: (H, W) → (H, W, 1)
        fracture = fracture.unsqueeze(-1)
        gradient = gradient.unsqueeze(-1)

        return fracture, gradient

    def generate_fracture(self, size, fracture_width_range):
        height, width = size
        fracture = torch.zeros((height, width), dtype=torch.float32)

        fracture_width = random.randint(*fracture_width_range)

        col = random.randint(0, width - fracture_width)
        fracture[:, col:col + fracture_width] = 1

        return fracture

    def generate_linear_gradient(self, fracture):
        gradient = torch.zeros_like(fracture)
        count = fracture.sum().int().item()

        if count > 0:
            # Linear gradient between 2 and 1
            values = torch.linspace(2, 1, steps=count)
            gradient[fracture == 1] = values

        return gradient


train_dataset = FractureGradientDataset(num_samples=10)
test_dataset = FractureGradientDataset(
    num_samples=5, size=(512, 512))  # example values for size

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Training the model
modes1, modes2, width = 4, 4, 64  # example values for modes and width
num_layers = 4
model = FNO2d(modes1=modes1, modes2=modes2, width=width, num_layers=num_layers)

# Train the model using PyTorch Lightning Trainer
# Set gpus=1 if you want to use GPU
trainer = pl.Trainer(max_epochs=100, accelerator='auto')
trainer.fit(model, train_loader)

test_results = trainer.test(model, dataloaders=test_loader)
# # Initialize lists to collect ground truth and predictions
# all_y = []
# all_y_hat = []

# # Iterate through the test results (from each batch)
# for result in test_results:
#     y = result['y']  # Ground truth
#     y_hat = result['y_hat']  # Predictions

#     # Append to lists (if you're working with 2D images, for example)
#     all_y.append(y)
#     all_y_hat.append(y_hat)

# # Convert lists to tensors (if you need to work with them as a single tensor)
# all_y = torch.cat(all_y, dim=0)
# all_y_hat = torch.cat(all_y_hat, dim=0)

# # For visualization, let's plot the first sample's ground truth and prediction
# plt.figure(figsize=(12, 6))

# # Plot the ground truth (first sample)
# plt.subplot(1, 2, 1)
# plt.imshow(all_y[0].cpu().numpy(), cmap='gray')
# plt.title('Ground Truth')

# # Plot the prediction (first sample)
# plt.subplot(1, 2, 2)
# plt.imshow(all_y_hat[0].cpu().detach().numpy(), cmap='gray')
# plt.title('Prediction')

# plt.show()

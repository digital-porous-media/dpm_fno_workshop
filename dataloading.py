import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

torch.random.manual_seed(13072894)


def compute_velocity(p, kappa, phi=0.1, dx=1.0, dy=1.0):
    # Compute gradients of pressure
    # Note: gradient returns (axis 0, axis 1)
    p = np.squeeze(p)
    dp_dy, dp_dx = np.gradient(p, dy, dx)

    # Ensure proper shape broadcasting
    kappa = np.asarray(kappa)
    phi = np.asarray(phi)

    # Apply Darcy’s law
    vx = -kappa / phi * dp_dx
    vy = -kappa / phi * dp_dy

    return vx, vy


class DarcyDataset(Dataset):
    def __init__(self, h5_path, resolution, dtype=torch.float32):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            resolution (str): Resolution group to load (e.g., 'resolution_64').
            dtype (torch.dtype): The data type to load the dataset as (default: torch.float32).
        """
        super().__init__()

        # Open the HDF5 file
        self.file = h5py.File(h5_path, 'r')
        # Get the resolution group
        self.resolution_group = self.file[resolution]
        self.perm = self.resolution_group['permeability']
        self.pressure = self.resolution_group['pressure']
        self.dtype = dtype

    def __len__(self):
        # The number of samples is the first dimension of the permeability dataset
        return self.perm.shape[0]

    def __getitem__(self, idx):
        # Load permeability and pressure samples for this index
        x = torch.tensor(self.perm[idx], dtype=self.dtype)
        y = torch.tensor(self.pressure[idx], dtype=self.dtype)

        # Add channel dimension (1, H, W) if needed, assuming it's 2D
        x = x.unsqueeze(-1)  # (H, W) -> (H, W, 1)
        y = y.unsqueeze(-1)  # (H, W) -> (H, W, 1)

        return x, y


def darcy_dataloader(h5_path, resolution, batch_size=16, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the given HDF5 file and resolution group.

    Args:
        h5_path (str): Path to the HDF5 file.
        resolution (str): Resolution group to load (e.g., 'resolution_64').
        batch_size (int): Batch size (default: 16).
        shuffle (bool): Whether to shuffle data (default: True).
        num_workers (int): Number of worker threads for data loading (default: 4).

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = DarcyDataset(h5_path, resolution)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MCFDataset(Dataset):
    def __init__(self, image_ids, data_path, t_in, t_out):
        #self.file = h5py.File(data_path, 'r')
        self.image_ids = image_ids
        self.data_path = data_path
        #self.t_input = self.file['t_in']
        #self.t_output = self.file['t_out']
        self.t_in = t_in
        self.t_out = t_out

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        # Load data from .h5py files
        self.File = h5py.File(self.data_path, "r")

        t_input = self.File["t_in"]
        t_output = self.File["t_out"]
        
        #input_field = self.file['t_in'][self.image_ids[idx]]
        input_field = torch.tensor(t_input[idx], dtype=torch.float32)
        input_field = torch.permute(input_field, (1, 2, 0))
        input_field, _, _ = self.z_score_normalize(input_field)

        #input_field = torch.from_numpy(input_field).float()
        input_field = input_field.reshape(128, 128, 1, self.t_in).repeat([1, 1, self.t_out, 1])
        # print(input_field.shape)

        # Target fields
        output_field = torch.tensor(t_output[idx], dtype=torch.float32)
        output_field = torch.permute(output_field, (1, 2, 0))
        output_field, original_means, original_stds = self.z_score_normalize(output_field)
        # print(output_field.shape)

        #output_field = torch.from_numpy(output_field).float()
        #original_means = torch.tensor(original_means)
        #original_stds = torch.tensor(original_stds)
        # print('Normal Target', output_field.min(), output_field.max())
        return input_field, output_field, original_means, original_stds

    def z_score_normalize(self, data):
        """
        Perform Z-score normalization along the last dimension (seq_len).
        Args:
            data (torch.tensor): Input data of shape [height, width, seq_len].
        Returns:
            normalized_data (torch.tensor): Normalized data of shape [height, width, seq_len].
            means (torch.tensor): Means along the seq_len dimension, shape [height, width].
            stds (torch.tensor): Standard deviations along the seq_len dimension, shape [height, width].
        """
        means = torch.mean(data, dim=-1, keepdims=True)  # Shape: [height, width, 1]
        stds = torch.std(data, dim=-1, keepdims=True)  # Shape: [height, width, 1]
        stds[stds == 0] = 1e-8
        normalized_data = (data - means) / stds
        return normalized_data, means, stds

    def normalize_to_11(self, data: np.ndarray):
        """
        Perform normalization along the last dimension (seq_len) to range [-1, 1].

        Args:
            data (np.ndarray): Input data of shape [height, width, seq_len].

        Returns:
            normalized_data (np.ndarray): Normalized data of shape [height, width, seq_len].
            mins (np.ndarray): Minimum values along seq_len, shape [height, width, 1].
            maxs (np.ndarray): Maximum values along seq_len, shape [height, width, 1].
        """
        mins = np.min(data, axis=-1, keepdims=True)
        maxs = np.max(data, axis=-1, keepdims=True)

        # Avoid division by zero
        denom = maxs - mins
        denom[denom == 0] = 1

        normalized_data = 2 * (data - mins) / denom - 1

        return normalized_data, mins, maxs

    def normalize_to_01(self, data: np.ndarray):
        """
        Perform normalization along the last dimension (seq_len) to range [0, 1].

        Args:
            data (np.ndarray): Input data of shape [height, width, seq_len].

        Returns:
            normalized_data (np.ndarray): Normalized data of shape [height, width, seq_len].
            mins (np.ndarray): Minimum values along seq_len, shape [height, width, 1].
            maxs (np.ndarray): Maximum values along seq_len, shape [height, width, 1].
        """
        mins = np.min(data, axis=-1, keepdims=True)
        maxs = np.max(data, axis=-1, keepdims=True)

        # Avoid division by zero
        denom = maxs - mins
        denom[denom == 0] = 1

        normalized_data = (data - mins) / denom

        return normalized_data, mins, maxs


def mcf_dataloader(image_ids, data_path, t_in, t_out, split, batch=1, num_workers=4, seed=1261613, **kwargs):
    dataset = MCFDataset(image_ids=image_ids, data_path=data_path, t_in=t_in, t_out=t_out)
    generator = torch.Generator().manual_seed(seed)
    assert len(split) == 3, "Split must be a list of length 3."
    assert round(sum(split), 6) == 1., "Sum of split must equal one."
    train_set, val_set, test_set = random_split(dataset, split, generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, persistent_workers=True,
                              num_workers=num_workers, **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, persistent_workers=True, num_workers=num_workers,
                            **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=num_workers, **kwargs)

    return train_loader, val_loader, test_loader


def split_indices(indices, split, seed=None):
    if seed is not None:
        np.random.seed(seed)

    assert len(split) == 3, "Split must be a list of length 3."
    assert round(sum(split), 6) == 1., "Sum of split must equal one."

    np.random.shuffle(indices)
    train_size = int(split[0] * len(indices))
    val_size = int(split[1] * len(indices))

    train_ids = indices[:train_size]
    val_ids = indices[train_size: (val_size + train_size)]
    test_ids = indices[(val_size + train_size):]

    return train_ids, val_ids, test_ids

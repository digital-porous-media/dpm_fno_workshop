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
        self.image_ids = image_ids
        self.data_path = data_path
        self.t_in = t_in
        self.t_out = t_out

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        try:
            self.file = h5py.File(self.data_path, 'r')
            # Load data from .h5py files
            input_field = self.file['t_in'][self.image_ids[idx]]
            input_field = np.transpose(input_field, (1, 2, 0))
            input_field, _, _ = self.z_score_normalize(input_field)

            input_field = torch.from_numpy(input_field).float()
            input_field = input_field.reshape(128, 128, 1, self.t_in).repeat([1, 1, self.t_out, 1])
            # print(input_field.shape)

            # Target fields
            output_field = self.file['t_out'][self.image_ids[idx]]
            output_field = np.transpose(output_field, (1, 2, 0))
            output_field, original_means, original_stds = self.z_score_normalize(output_field)
            # print(output_field.shape)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {e} not found.")

        output_field = torch.from_numpy(output_field).float()
        original_means = torch.tensor(original_means)
        original_stds = torch.tensor(original_stds)
        # print('Normal Target', output_field.min(), output_field.max())
        return input_field, output_field, original_means, original_stds

    def z_score_normalize(self, data):
        """
        Perform Z-score normalization along the last dimension (seq_len).
        Args:
            data (np.ndarray): Input data of shape [height, width, seq_len].
        Returns:
            normalized_data (np.ndarray): Normalized data of shape [height, width, seq_len].
            means (np.ndarray): Means along the seq_len dimension, shape [height, width].
            stds (np.ndarray): Standard deviations along the seq_len dimension, shape [height, width].
        """
        means = np.mean(data, axis=-1, keepdims=True)  # Shape: [height, width, 1]
        stds = np.std(data, axis=-1, keepdims=True)  # Shape: [height, width, 1]
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


if __name__ == '__main__':
    # Assuming the FNOHDF5Dataset is already defined
    dataset = DarcyDataset('darcy_data.h5', resolution='resolution_128')

    # Specify the sizes of your splits
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.15 * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - \
        val_size  # Remaining 15% for testing

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Extract a sample at a specific index (e.g., index 5)
    # k, p = dataset_64[0]

    # Now you can plot or use this sample
    # plot_sample(x_sample, y_sample, idx=5)

    # n64_trainloader = darcy_dataloader(
    #     'darcy_data.h5', 'resolution_64', batch_size=16)

    # plot_sample_idx(n64_trainloader, idx=5)
    # with h5py.File('darcy_data.h5', 'r') as h5f:
    # k = h5f['resolution_64']['permeability'][0]
    # p = h5f['resolution_64']['pressure'][0]
    # fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # ax[0].imshow(k.squeeze(), cmap='viridis')
    # ax[0].invert_yaxis()
    # ax[1].imshow(p.squeeze(), cmap='plasma')
    # ax[1].contour(p.squeeze(), levels=10, colors='k')
    # ax[1].invert_yaxis()
    # vx, vy = compute_velocity(p.squeeze(), k.squeeze())

    # # vel = np.load(
    # #     "D:/fno_workshop_training_data/N64/v_output/sample_0.npz")
    # # vx = vel['vx']
    # # vy = vel['vy']
    # # print(vx.shape, vy.shape)
    # magnitude = np.sqrt(vx**2 + vy**2)
    # X, Y = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]))

    # # ax[2].quiver(X, Y, vx, vy, magnitude, cmap='plasma',
    # #              units='xy', scale_units='xy', scale=0.85)
    # ax[2].imshow(magnitude, cmap='plasma')
    # # ax[2].contour(magnitude, levels=10, colors='k')
    # ax[2].invert_yaxis()
    # plt.show()

import numpy as np
import os
from glob import glob
import h5py
def read_raw_file(filepath, shape, dtype=np.float32):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
        return data.reshape(shape)


# Set your paths
path = '/scratch/08780/cedar996/lbfoam/level_set/training_data'

array_shape1 = (8, 128, 128)
array_shape2 = (16, 128, 128)
# Sorted list of file paths
files1 = sorted(glob(os.path.join(path, 'input*.raw')))
files2 = sorted(glob(os.path.join(path, 'target*.raw')))
# Sanity check
assert len(files1) == 160 and len(files2) == 160, "Check file counts!"
assert all(os.path.basename(f1)[-14:-9] == os.path.basename(f2)[-16:-11] for f1, f2 in zip(files1, files2)), "Order mismatch!"

# Load and stack
dataset1 = np.stack([read_raw_file(f, array_shape1) for f in files1], axis=0)
dataset2 = np.stack([read_raw_file(f, array_shape2) for f in files2], axis=0)

print("Dataset1 shape:", dataset1.shape)  # (160, 8, 128, 128)
print("Dataset2 shape:", dataset2.shape)
with h5py.File('mc_flow_data.h5', 'w') as f:
    f.create_dataset('t_in', data=dataset1)
    f.create_dataset('t_out', data=dataset2)

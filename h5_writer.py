import os
import numpy as np
import h5py
from tqdm import tqdm


def convert_to_grouped_hdf5(data_root, output_file, dtype=np.float32):
    """
    Converts data under training_data/Nxx/k_input and p_output folders
    into a single HDF5 file with resolution-specific groups.

    Args:
        data_root (str): Path to the parent data directory (e.g. 'training_data').
        output_file (str): Path to the output HDF5 file.
        dtype (np.dtype): Data type for storage (e.g., np.float32).
    """
    resolutions = sorted([d for d in os.listdir(
        data_root) if os.path.isdir(os.path.join(data_root, d))])

    with h5py.File(output_file, 'w') as h5f:
        for res in resolutions:
            res_path = os.path.join(data_root, res)
            k_dir = os.path.join(res_path, 'k_input')
            p_dir = os.path.join(res_path, 'p_output')

            if not (os.path.exists(k_dir) and os.path.exists(p_dir)):
                print(f"\u26A0 Skipping {res} — missing k_input or p_output.")
                continue

            print(f"\U0001F501 Processing resolution: {res}")
            # convert "N64" → "resolution_64"
            res_key = f"resolution_{res[1:]}"

            permeability_list = []
            pressure_list = []

            filenames = sorted([f for f in os.listdir(
                k_dir) if f.startswith('sample_') and f.endswith('.npy')])

            for fname in tqdm(filenames, desc=f"  → {res}"):
                k_path = os.path.join(k_dir, fname)
                p_path = os.path.join(p_dir, fname)

                if not os.path.exists(p_path):
                    print(
                        f"    \u26A0 Missing pressure file for {fname}, skipping.")
                    continue

                k_field = np.load(k_path).astype(dtype)
                p_field = np.load(p_path).astype(dtype)

                permeability_list.append(k_field)
                pressure_list.append(p_field)

            # Stack and create group
            permeability_array = np.stack(permeability_list, axis=0)
            pressure_array = np.stack(pressure_list, axis=0)

            grp = h5f.create_group(res_key)
            grp.create_dataset('permeability', data=permeability_array)
            grp.create_dataset('pressure', data=pressure_array)

            print(
                f"\u2705 Saved {len(permeability_array)} samples under group '{res_key}'")

    print(f"\n\U0001F389 All resolutions saved to {output_file}")


if __name__ == "__main__":
    convert_to_grouped_hdf5(
        data_root='D:/fno_workshop_training_data/large_res',
        output_file='darcy_data_res.h5'
    )

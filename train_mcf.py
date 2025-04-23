import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from FNO3d import FNO3D
from dataloading import MCFDataset, mcf_dataloader
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import glob
from pathlib import Path
import json


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    hparams = {
        'net_name': "workshop_t123",
        'learning_rate': 1e-3,
        'batch_size': 1,
        'epochs': 200,
        'val_interval': 5,
        'modes1': 12,
        'modes2': 12,
        'modes3': 3,
        'width': 24,
        'beta_1': 1,
        'beta_2': 1,
        'T_in': 8,
        'T_out': 16,
        'seed': 189031465,
        'model': 'tFNO3DModel',
        'patience': 9999
    }

    # Load the data
    seed = 189031465

    n_samples = 160
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=159, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = mcf_dataloader(image_ids,
                                                                       data_path="/scratch/08780/cedar996/lbfoam/level_set/mc_flow_data.h5",
                                                                       t_in=hparams['T_in'],
                                                                       t_out=hparams['T_out'],
                                                                       seed=seed,
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)

    # Try loading a model first
    try:
        model_dir = f"lightning_logs/{hparams['net_name']}/checkpoints"
        model_loc = glob.glob(f'{model_dir}/*val*.ckpt')[0]
        print(f'Loading {model_loc}')
        with open(f"lightning_logs/{hparams['net_name']}/hparam_config.json", 'r') as f:
            json_string = f.read()

        hparams = json.loads(json_string)

        hparams['seed'] = 189031465
        np.random.seed(hparams['seed'])
        model = FNO3D.load_from_checkpoint(model_loc,
                                           model=hparams['model'],
                                           in_channels=hparams['T_in'],
                                           out_channels=1,
                                           modes1=hparams['modes1'],
                                           modes2=hparams['modes2'],
                                           modes3=hparams['modes3'],
                                           width=hparams['width'],
                                           beta_1=hparams['beta_1'],
                                           beta_2=hparams['beta_2'],
                                           lr=hparams['learning_rate'],
                                           )

    except IndexError:
        # Instantiate the model
        print('Instantiating a new model...')
        model = FNO3D(net_name=hparams['net_name'],
                      in_channels=hparams['T_in'],
                      out_channels=1,
                      modes1=hparams['modes1'],
                      modes2=hparams['modes2'],
                      modes3=hparams['modes3'],
                      width=hparams['width'],
                      beta_1=hparams['beta_1'],
                      beta_2=hparams['beta_2'],
                      model=hparams['model'],
                      lr=hparams['learning_rate'],)
        log_path = Path(f"./lightning_logs/{hparams['net_name']}")
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path / "hparam_config.json", 'w') as f:
            json.dump(hparams, f)

    # Add some checkpointing callbacks
    cbs = [ModelCheckpoint(monitor="loss", filename="{epoch:02d}-{loss:.2f}",
                           save_top_k=1,
                           mode="min"),
           ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}-{val_loss:.2f}",
                           save_top_k=1,
                           mode="min"),
           EarlyStopping(monitor="val_loss", check_finite=False, patience=hparams['patience'])]

    trainer = pl.Trainer(
        #strategy='ddp_find_unused_parameters_true',
        callbacks=cbs,  # Add the checkpoint callback
        max_epochs=hparams['epochs'],
        check_val_every_n_epoch=hparams['val_interval'],
        log_every_n_steps=n_samples * split[0],#/3,
    )

    trainer.fit(model, train_dataloader, val_dataloader)



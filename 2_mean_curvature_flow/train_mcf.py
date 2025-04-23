import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from network.FNO3d import FNO3D

from dataloading import MCFDataset, mcf_dataloader
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import glob
from pathlib import Path
import json


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    cfg = OmegaConf.load("config.yaml")

    # Load the data
    seed = 189031465

    n_samples = 160
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=159, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = mcf_dataloader(image_ids,
                                                                       data_path="/scratch/08780/cedar996/lbfoam/level_set/mc_flow_data.h5",
                                                                       t_in=cfg.T_in,
                                                                       t_out=cfg.T_out,
                                                                       seed=seed,
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)

    # Instantiate the model
    print('Instantiating a new model...')
    model = FNO3D(net_name=cfg.net_name,
                  in_channels=cfg.T_in,
                  out_channels=1,
                  modes1=cfg.modes1,
                  modes2=cfg.modes2,
                  modes3=cfg.modes3,
                  width=cfg.width,
                  lr=cfg.lr)


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



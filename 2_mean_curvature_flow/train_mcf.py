import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from network.FNO3d import FNO3D

from dataloading import MCFDataset, mcf_dataloader
from torch.utils.data import DataLoader, random_split
# from mcf_dataloading import MCFDataset
from omegaconf import OmegaConf

import torch
import numpy as np



if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    cfg = OmegaConf.load("config.yaml")

    # Load the data
    seed = cfg.seed
    # dataset = MCFDataset("mc_flow_data.h5")
    n_samples = 160 # len(dataset)

    # Train size, val size, test size
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=159, size=(n_samples,))
    train_loader, val_loader, test_loader = mcf_dataloader(image_ids,
                                                           data_path="mc_flow_data.h5",
                                                           t_in=cfg.T_in,
                                                           t_out=cfg.T_out,
                                                           seed=cfg.seed,
                                                           split=split,
                                                           num_workers=8,
                                                           pin_memory=True,
                                                           )

    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, split)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Instantiate the model
    print('Instantiating a new model...')
    model = FNO3D(net_name=cfg.net_name,
                  in_channels=cfg.T_in,
                  out_channels=1,
                  modes1=cfg.modes1,
                  modes2=cfg.modes2,
                  modes3=cfg.modes3,
                  width=cfg.width,
                  lr=cfg.learning_rate)

    # Add some checkpointing callbacks
    cbs = [ModelCheckpoint(monitor="loss", filename="{epoch:02d}-{loss:.2f}",
                           save_top_k=1,
                           mode="min"),
           ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}-{val_loss:.2f}",
                           save_top_k=1,
                           mode="min")]

    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator='auto',
                         callbacks=cbs,
                         check_val_every_n_epoch=cfg.val_interval,
                         log_every_n_steps=n_samples * split[0],
                         )

    trainer.fit(model, train_loader, val_loader)

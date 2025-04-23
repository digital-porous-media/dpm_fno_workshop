import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from dataloading import DarcyDataset

from torch.utils.data import DataLoader, random_split
import lightning as pl

from network.FNO2d import FNO2D

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")

    pl.seed_everything(cfg.seed)

    # Load the Darcy dataset
    dataset = DarcyDataset('darcy_data_res.h5', resolution='resolution_64')

    # Specify the sizes of your splits
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.15 * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - \
        val_size  # Remaining 15% for testing

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=1, shuffle=False)

    # Instantiate an FNO2d model
    model = FNO2D(net_name=cfg.net_name,
                  modes1=cfg.modes1,
                  modes2=cfg.modes2,
                  width=cfg.width,
                  num_layers=cfg.n_layers,
                  lr=cfg.learning_rate)

    # Train the model using Lightning trainer
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         strategy='auto',
                         accumulate_grad_batches=cfg.batch_size,
                         enable_checkpointing=True)
    trainer.fit(model, train_loader)

    # test_results = trainer.test(model, dataloaders=test_loader)


    # res_128_dataset = DarcyDataset(
    #     'darcy_data.h5', resolution='resolution_128')
    # _, _, res_128_dataset = random_split(
    #     res_128_dataset, [80, 10, 10])
    # res_128_test_loader = DataLoader(res_128_dataset, batch_size=1, shuffle=False)
    # test_results = trainer.test(model, dataloaders=res_128_test_loader)
    # _, _, test_dataset = random_split(
    #     dataset, [train_size, val_size, test_size])
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

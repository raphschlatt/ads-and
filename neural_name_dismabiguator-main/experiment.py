from AND_nn_exp2 import ANDismabiguator                  # Import model class
from AND_readdata_exp2 import PairsANDDataModule         # Import custom DataModule for loading data
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from pytorch_lightning.callbacks import EarlyStopping

# Set random seed for reproducibility based on the first command-line argument
seed_everything(int(sys.argv[1]))

from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration object for experiment setup.

    Attributes:
        save_dir (str): Directory where training logs and CSV outputs will be stored.
    """
    save_dir: str = "logs/"

config = Config()

# Initialize the custom DataModule that handles train/val data loading
data_module = PairsANDDataModule(batch_size=2048, seed=int(sys.argv[1]))

# Prepare dataloaders
data_module.setup(stage='fit')
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

def run_experiment(number):
    """
    Runs a single training experiment using the specified run number.

    Args:
        number (int): Identifier for the experiment run; used to label checkpoints and logs.
    """
    # File naming format for checkpoints
    filename_template = f"{number}_specter_chars2vec" + "{epoch}-{val_loss:.2f}"

    # Callback: Saves model checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",                        # Monitors validation loss to determine the best model
        mode="min",
        save_weights_only=True,
        dirpath='/specter_chars2vec',
        filename=filename_template,
        save_last=True,                            # Also save the last epoch checkpoint
        every_n_epochs=1
    )

    # Callback: Stops training if validation loss doesn’t improve for 25 epochs
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=25,
        verbose=True
    )

    # PyTorch Lightning trainer setup
    trainer = pl.Trainer(
        accelerator='gpu',                         # Train on GPU
        devices=1,
        max_epochs=250,
        logger=CSVLogger(save_dir=config.save_dir, name=f"{number}_specter_chars2vec.log"),  # Save training logs to CSV
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping]
    )

    # Initialize model with run number (used for saving best thresholds)
    model = ANDismabiguator(run_num=int(sys.argv[1]))

    # Start training
    trainer.fit(model, train_loader, val_loader)
    

# Run the experiment when executed via command line
if __name__ == "__main__":
    run_experiment(int(sys.argv[1]))
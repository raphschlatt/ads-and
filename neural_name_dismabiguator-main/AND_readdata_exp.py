import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
import random
from lightning.pytorch import seed_everything

# ----------------------------
# Load data into memory
# ----------------------------

DATA_DIR = "path/to/data/"

# Load labeled pairs
author_references = pd.read_json(f"{DATA_DIR}/author_pair_labels.json")

# Load embeddings for the two authors in each pair
author1_embed = np.load(f"{DATA_DIR}/author1_embeddings.npy")   # Embeddings for author 1
author2_embed = np.load(f"{DATA_DIR}/author2_embeddings.npy")   # Embeddings for author 2

# Extract train/validation/test indices based on the split column
train_indices = author_references.query('split == "train"').index.values
val_indices = author_references.query('split == "val"').index.values
test_indices = author_references.query('split == "test"').index.values


# ----------------------------
# Dataset that loads everything at once (no chunking)
# ----------------------------

class NonChunkedDataset(Dataset):
    def __init__(self, indices, shuffle=False):
        """
        A simple dataset that loads all pairs into memory at once.
        Each item contains (author1_embedding, author2_embedding) and a label.
        """
        self.indices = indices
        
        # Load embeddings into tensors for the selected indices
        self.author1_data = torch.tensor(author1_embed[indices], dtype=torch.float32)
        self.author2_data = torch.tensor(author2_embed[indices], dtype=torch.float32)
        
        # Load labels for those indices (True/False for match or no-match)
        self.labels = torch.tensor(
            author_references.label.iloc[indices].values, 
            dtype=torch.bool
        )
        
        # Combine both author embeddings into one tensor of shape [N, 2, embedding_dim]
        self.data = torch.stack([self.author1_data, self.author2_data], dim=1)
        
        # Free unused references now that stacked data is created
        del self.author1_data
        del self.author2_data

    def __len__(self):
        # Total number of samples in the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Return one pair and its label
        return self.data[idx], self.labels[idx]


# ----------------------------
# Lightning DataModule to organize data loaders
# ----------------------------

class PairsANDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 10000, seed = 0):
        """
        DataModule to provide train/val/test data loaders.
        Sets a global seed for reproducibility.
        """
        super().__init__()
        seed_everything(seed)  # Ensures reproducible data order
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        """
        Called by Lightning to set up datasets for the given stage.
        """
        # For training or full setup, create train and validation datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = NonChunkedDataset(train_indices, shuffle=True)
            self.val_dataset = NonChunkedDataset(val_indices, shuffle=False)
        
        # For testing or full setup, create test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = NonChunkedDataset(test_indices, shuffle=False)

    def train_dataloader(self):
        # DataLoader for training with shuffling for randomness
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=3,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        # DataLoader for validation (no shuffle)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        # DataLoader for testing (no shuffle)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )

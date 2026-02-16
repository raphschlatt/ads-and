import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import os
from torchmetrics.classification import BinaryAccuracy, BinaryROC
from torchmetrics import Precision, Recall
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ANDismabiguator(pl.LightningModule):
    
    def __init__(self, run_num, dropout_prob=0.2):
        """
        Neural network model for author name disambiguation using cosine similarity-based training.
        Tracks run number for saving threshold results.
        """
        super().__init__()
        self.run_num = run_num
        
        # Neural network encoder maps each author embedding to a learned representation
        self.l1 = nn.Sequential(
            nn.Linear(818, 1024),   # Input layer
            nn.SELU(),               # Nonlinear activation
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),   # Hidden layer
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 256),    # Output embedding size
            nn.SELU()
        )

        # Loss parameters and metrics
        self.margin = 0.65
        self.accuracy = BinaryAccuracy(threshold = 0.65)
        self.BCELoss = nn.BCELoss()
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss(margin=self.margin)
        self.TripletMarginWithDistanceLoss = nn.TripletMarginWithDistanceLoss(
            distance_function=nn.CosineSimilarity(), 
            margin=self.margin
        )

        # Buffers to store outputs for epoch-end computation
        self.trainoutputs = []
        self.outputs = []

        # Validation metrics
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')

        # Store thresholds found across epochs
        self.val_CS = []
        self.val_y = []
        self.optimal_thresholds = []

    def forward(self, x):
        """Forward pass through the encoder network."""
        return self.l1(x)

    def ANDloss(self, batch):
        """
        Computes CosineEmbeddingLoss for pairwise training.
        Used primarily during testing.
        """
        x, y = batch
        y_hat = self(x)

        # Convert labels to +1 / -1 for CosineEmbeddingLoss
        sign = torch.where(y, torch.tensor(1), torch.tensor(-1))

        # Cosine similarity between embeddings
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])

        # Main loss
        loss = self.CosineEmbeddingLoss(y_hat[:, 0, :], y_hat[:, 1, :], sign)
        accuracy = self.accuracy(torch.clamp(cosine_sim, min=0, max=1), y)
        
        # Save for epoch metrics
        self.trainoutputs.append({'y': y, 'CS': cosine_sim})
        return accuracy, loss.mean()

    def nceloss(self, batch):
        """
        NT-Xent (InfoNCE) loss for binary pairs.
        Encourages positive pairs to have higher cosine similarity than negatives.
        """
        temperature = 0.25
        x, y = batch
        y_hat = self(x)

        # Compute cosine similarity between pair embeddings
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])

        # Only positives used for numerator
        positive_indices = torch.where(y == 1)[0]
        cosine_sim = cosine_sim / temperature

        # NT-Xent loss
        nll = -cosine_sim[positive_indices] + torch.logsumexp(cosine_sim, dim=-1)
        loss = nll.mean()

        accuracy = self.accuracy(torch.clamp(cosine_sim, min=0, max=1), y)

        if self.training:
            self.trainoutputs.append({'y': y, 'CS': cosine_sim})

        return accuracy, loss

    def triplet_loss(self, batch):
        """
        Triplet loss using cosine similarity as distance metric.
        Creates anchor-positive-negative triplets from batch.
        """
        x, y = batch
        y_hat = self(x)

        sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        positive_indices = torch.where(y == 1)[0]
        negative_indices = torch.where(y == 0)[0]

        # Match equal number of positives and negatives
        num_triplets = min(len(positive_indices), len(negative_indices))
        scramble_indices = np.random.randint(0, num_triplets, size=num_triplets)

        # Form triplets
        anchor_embeddings = y_hat[positive_indices[:num_triplets]][:, 0, :]
        positive_embeddings = y_hat[positive_indices[:num_triplets]][:, 1, :]
        negative_embeddings = y_hat[scramble_indices[:num_triplets]][:, 1, :]
        
        loss = self.TripletMarginWithDistanceLoss(anchor_embeddings, positive_embeddings, negative_embeddings)
        accuracy = self.accuracy(torch.clamp(sim, min=0, max=1), y)

        if self.training:
            self.trainoutputs.append({'y': y, 'CS': sim})

        return accuracy, loss

    def training_step(self, batch, batch_idx):
        """Runs one training step using NCE loss."""
        accuracy, loss = self.nceloss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs one validation step."""
        x, y = batch
        y_hat = self(x)
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        accuracy, loss = self.nceloss(batch)

        # Log batch-level val metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        # Store to determine best threshold later
        self.outputs.append({'y': y, 'CS': cosine_sim})
        return loss

    def on_train_epoch_end(self):
        """
        At the end of each train epoch:
         - Aggregate outputs
         - Compute ROC
         - Determine optimal decision threshold (G-mean)
         - Log the best threshold and accuracy at threshold
        """
        y, CS = [], []
        for outputs in self.trainoutputs[-1:]:
            y.append(outputs['y'])
            CS.append(outputs['CS'])

        targets = torch.stack(y).flatten(0, 1)
        preds = torch.stack(CS).flatten(0, 1)
        self.trainoutputs.clear()

        roc = BinaryROC()
        fpr, tpr, thresholds = roc(preds, targets)

        # Use G-mean to find best threshold
        gmeans = torch.sqrt(tpr * (1 - fpr))
        index = torch.argmax(gmeans)
        optimal_threshold = thresholds[index]

        self.log("train_end_acc", tpr[index], prog_bar=True, sync_dist=True)
        self.log("train_end_threshold", optimal_threshold.item(), prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Similar to train end:
         - Determine optimal threshold on validation data
         - Compute precision, recall, F1
         - Save threshold for this run
        """
        y, CS = [], []
        for outputs in self.outputs:
            y.append(outputs['y'])
            CS.append(outputs['CS'])

        targets = torch.cat(y).view(-1)
        preds = torch.cat(CS).view(-1)
        roc = BinaryROC()

        fpr, tpr, thresholds = roc(preds, torch.tensor(targets))
        gmeans = torch.sqrt(tpr * (1 - fpr))
        index = torch.argmax(gmeans)
        optimal_threshold = thresholds[index]

        self.outputs.clear()

        self.log("val_end_acc", tpr[index], prog_bar=True, sync_dist=True)
        self.log("val_end_threshold", optimal_threshold.item(), prog_bar=True, sync_dist=True)

        # Convert predictions to binary using threshold
        optimal_preds = torch.where(preds > thresholds[index], 1, 0).cpu()
        tensor_targets = torch.tensor(targets, dtype=torch.int)

        precision_mean = self.precision(optimal_preds, tensor_targets).item()
        recall_mean = self.recall(optimal_preds, tensor_targets).item()

        # Compute F1
        def f1score(p, r):
            return 0 if p + r == 0 else 2 * (p * r) / (p + r)

        f1_score = f1score(precision_mean, recall_mean)

        # Log val metrics
        self.log("val_precision", precision_mean, prog_bar=True, sync_dist=True)
        self.log("val_recall", recall_mean, prog_bar=True, sync_dist=True)
        self.log("val_f1_score", f1_score, prog_bar=True, sync_dist=True)

        self.optimal_thresholds.append(optimal_threshold.item())
        np.save(
            f'/specter_chars2vec_nce_vtest/{self.run_num}',
            np.array(self.optimal_thresholds)
        )
        
    def test_step(self, batch, batch_idx):
        """Evaluate on test set using CosineEmbeddingLoss."""
        accuracy, loss = self.ANDloss(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", accuracy, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Use Adam optimizer and reduce learning rate when validation loss plateaus.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.01,
                patience=25
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

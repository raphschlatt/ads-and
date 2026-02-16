# Import necessary modules and functions
import os  # Module for interacting with the operating system
import re  # Module for regular expressions

import numpy as np  # NumPy for numerical operations
import torch  # PyTorch for deep learning
from torch.nn import (
    functional as F,
)  # Functional module from PyTorch for common functions

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)  # Metrics for evaluation

from AND_nn_exp import ANDismabiguator  # Import the ANDismabiguator model
from AND_readdata_exp import (
    PairsANDDataModule2,
)  # Import the data module for loading pairs data

# Initialize the data module with specified batch and chunk sizes
data_module = PairsANDDataModule2(batch_size=2048, chunk_size=500000)


# Function to evaluate a single checkpoint
def evaluate_checkpoint(checkpoint_path, data_module):
    # Extract the run number and epoch number from the checkpoint path using regex
    number_match = re.search(r"/(\d+)_specter_chars2vec", checkpoint_path)
    epoch_match = re.search(r"epoch=(\d+)", checkpoint_path)

    if number_match and epoch_match:
        number = number_match.group(1)
        epoch_number = int(epoch_match.group(1))
    else:
        raise ValueError(
            "Required number or epoch number not found in the checkpoint path"
        )

    # Construct the path to the NumPy file containing thresholds for the specific run
    threshold_file_path = os.path.join(
        f"/specter_chars2vec/{number}.npy"
    )

    # Check if the NumPy file exists
    if not os.path.isfile(threshold_file_path):
        raise FileNotFoundError(f"Threshold file {threshold_file_path} not found")

    # Load the thresholds from the NumPy file
    thresholds = np.load(threshold_file_path)

    # Extract the relevant threshold based on the epoch number
    if epoch_number >= len(thresholds):
        raise IndexError(
            f"Epoch number {epoch_number} is out of range for the thresholds file"
        )

    threshold = thresholds[epoch_number]

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Restore the model state from the checkpoint
    model.load_state_dict(checkpoint["state_dict"])

    # Set the model to evaluation mode
    model.eval()

    # Load the evaluation data
    eval_data_loader = data_module.test_dataloader()

    all_labels = []
    all_preds = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for data in eval_data_loader:
            inputs, labels = data  # Adjust according to your data structure
            outputs = model(inputs)

            # Ensure outputs have the shape [batch_size, 2, feature_dim]
            output1 = outputs[:, 0, :]
            output2 = outputs[:, 1, :]

            # Calculate cosine similarity between the pairs
            cosine_sim = F.cosine_similarity(output1, output2, dim=1)

            # Convert cosine similarity to binary predictions using the threshold
            preds = (cosine_sim >= threshold).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


# List of checkpoint paths to evaluate
checkpoint_paths = [
    "/specter_chars2vec/1_specter_chars2vec_epoch=29-val_loss=0.02.ckpt",
    "/specter_chars2vec/2_specter_chars2vec_epoch=29-val_loss=0.02.ckpt",
    "/specter_chars2vec/3_specter_chars2vec_epoch=29-val_loss=0.02.ckpt",
    "/specter_chars2vec/4_specter_chars2vec_epoch=29-val_loss=0.02.ckpt",
    "/specter_chars2vec/5_specter_chars2vec_epoch=29-val_loss=0.02.ckpt",
    # Add more checkpoint paths as needed
]

# Initialize the model and set up the data module for testing
data_module.setup(stage="test")
model = ANDismabiguator()

# Evaluate all checkpoints and store metrics
all_accuracies, all_precisions, all_recalls, all_f1s = [], [], [], []

# Iterate over each checkpoint and evaluate it
for checkpoint_path in checkpoint_paths:
    accuracy, precision, recall, f1 = evaluate_checkpoint(checkpoint_path, data_module)
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

# Print out the metrics for each checkpoint
for i, checkpoint_path in enumerate(checkpoint_paths):
    print(f"Metrics for {checkpoint_path}:")
    print(f"Accuracy: {all_accuracies[i]}")
    print(f"Precision: {all_precisions[i]}")
    print(f"Recall: {all_recalls[i]}")
    print(f"F1 Score: {all_f1s[i]}")

# Calculate mean metrics
mean_accuracy = np.mean(all_accuracies)
mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
mean_f1 = np.mean(all_f1s)

# Calculate standard error of the mean (SEM) for each metric
sem_accuracy = np.std(all_accuracies) / np.sqrt(len(all_accuracies))
sem_precision = np.std(all_precisions) / np.sqrt(len(all_precisions))
sem_recall = np.std(all_recalls) / np.sqrt(len(all_recalls))
sem_f1 = np.std(all_f1s) / np.sqrt(len(all_f1s))

# Print mean and SEM for each metric
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {sem_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f} ± {sem_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f} ± {sem_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f} ± {sem_f1:.4f}")

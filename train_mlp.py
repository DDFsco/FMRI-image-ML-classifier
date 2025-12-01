"""
EECS 442 Final Project - BrainMapNet

Train fMRI MLP Baseline
    Train an MLP classifier on top activated voxels from fMRI data
    Usage: python train_mlp.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_train_val_test_datasets
from model.fmri_mlp import fMRIMLP
from train_common import count_parameters, evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import set_random_seed, make_training_plot
from torch.utils.data import DataLoader, TensorDataset


def extract_top_voxels(X: np.ndarray, top_k: int = 2000) -> np.ndarray:
    """Extract top k most activated voxels from each image.
    
    Args:
        X: Image array of shape (N, C, H, W)
        top_k: Number of top voxels to extract
        
    Returns:
        Array of shape (N, top_k) with top activated voxels
    """
    # Flatten spatial dimensions: (N, C, H, W) -> (N, C*H*W)
    N, C, H, W = X.shape
    X_flat = X.reshape(N, -1)
    
    # For each sample, get top k voxels by absolute value
    top_voxels = []
    for i in range(N):
        # Get absolute values and sort
        abs_values = np.abs(X_flat[i])
        top_indices = np.argsort(abs_values)[-top_k:]
        top_voxels.append(X_flat[i, top_indices])
    
    return np.array(top_voxels)


def get_mlp_loaders(data_root: str, batch_size: int, top_k: int = 2000, image_dim: int = 224):
    """Get DataLoaders for MLP training with top voxels."""
    tr, va, te, _ = get_train_val_test_datasets(
        data_root=data_root,
        image_dim=image_dim
    )
    
    # Extract images and labels
    print(f"Extracting top {top_k} voxels...")
    tr_images = []
    tr_labels = []
    for i in range(len(tr)):
        img, label = tr[i]
        tr_images.append(img.numpy())
        tr_labels.append(label.item())
    
    va_images = []
    va_labels = []
    for i in range(len(va)):
        img, label = va[i]
        va_images.append(img.numpy())
        va_labels.append(label.item())
    
    te_images = []
    te_labels = []
    for i in range(len(te)):
        img, label = te[i]
        te_images.append(img.numpy())
        te_labels.append(label.item())
    
    tr_X = np.array(tr_images)
    va_X = np.array(va_images)
    te_X = np.array(te_images)
    
    # Extract top voxels
    tr_X_voxels = extract_top_voxels(tr_X, top_k)
    va_X_voxels = extract_top_voxels(va_X, top_k)
    te_X_voxels = extract_top_voxels(te_X, top_k)
    
    # Create TensorDatasets
    tr_dataset = TensorDataset(
        torch.from_numpy(tr_X_voxels).float(),
        torch.from_numpy(np.array(tr_labels)).long()
    )
    va_dataset = TensorDataset(
        torch.from_numpy(va_X_voxels).float(),
        torch.from_numpy(np.array(va_labels)).long()
    )
    te_dataset = TensorDataset(
        torch.from_numpy(te_X_voxels).float(),
        torch.from_numpy(np.array(te_labels)).long()
    )
    
    # Create DataLoaders
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader


def main():
    """Train MLP and show training plots."""
    # Configuration
    data_root = "../dataset_combined_vertical"
    input_dim = 2000
    hidden_dims = [512, 256]
    dropout_rate = 0.3
    num_classes = 2
    learning_rate = 1e-3
    batch_size = 32
    image_dim = 224
    checkpoint_dir = "./checkpoints/fmri_mlp/"
    patience = 10
    
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader = get_mlp_loaders(
        data_root=data_root,
        batch_size=batch_size,
        top_k=input_dim,
        image_dim=image_dim
    )
    
    # Model
    model = fMRIMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading fMRI MLP...")
    model, start_epoch, stats = restore_checkpoint(model, checkpoint_dir)

    axes = make_training_plot(name="fMRI MLP Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
    )

    # Initial val loss for early stopping
    prev_val_loss = stats[0][1] if stats else float('inf')

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    curr_count_to_patience = 0
    
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, checkpoint_dir, stats)

        # Update early stopping parameters
        curr_count_to_patience, prev_val_loss = early_stopping(stats, curr_count_to_patience, prev_val_loss)

        epoch += 1
    
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig(f"fmri_mlp_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()


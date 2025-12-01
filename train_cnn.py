"""
EECS 442 Final Project - BrainMapNet

Train fMRI CNN
    Train a CNN classifier (ResNet-18 or MobileNet) on fMRI activation maps
    Usage: python train_cnn.py
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_train_val_test_loaders
from model.fmri_cnn import fMRICNN
from train_common import count_parameters, evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import set_random_seed, make_training_plot


def main():
    """Train CNN and show training plots."""
    # Configuration
    data_root = "../dataset_combined_vertical"
    architecture = "resnet18"  # or "mobilenet"
    pretrained = True
    freeze_backbone = False
    num_classes = 2
    learning_rate = 1e-4
    batch_size = 16
    image_dim = 224
    checkpoint_dir = "./checkpoints/fmri_cnn/"
    patience = 10
    
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader = get_train_val_test_loaders(
        data_root=data_root,
        batch_size=batch_size,
        image_dim=image_dim
    )
    
    # Model
    model = fMRICNN(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Architecture: {architecture}")
    print(f"Pretrained: {pretrained}")
    print(f"Freeze backbone: {freeze_backbone}")
    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading fMRI CNN...")
    model, start_epoch, stats = restore_checkpoint(model, checkpoint_dir)

    axes = make_training_plot(name=f"fMRI CNN Training ({architecture})")

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
    plt.savefig(f"fmri_cnn_training_plot_{architecture}_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()


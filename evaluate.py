"""
EECS 442 Final Project - BrainMapNet

Evaluate fMRI Models
    Evaluate trained models and generate confusion matrix and Grad-CAM visualizations
    Usage: python evaluate.py --model cnn --architecture resnet18
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from torch.nn.functional import softmax
import seaborn as sns
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_train_val_test_loaders, get_train_val_test_datasets
from model.fmri_cnn import fMRICNN
from model.fmri_mlp import fMRIMLP
from train_common import restore_checkpoint, predictions
from utils import set_random_seed

# Grad-CAM is optional (only needed for visualization)
try:
    from gradcam import visualize_gradcam
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: opencv-python not installed. Grad-CAM visualization will not be available.")
    print("Install with: pip install opencv-python")


def evaluate_model(model, loader, criterion, device='cpu'):
    """Evaluate model on a dataset."""
    model.eval()
    y_true, y_pred, y_score = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            y_true.append(y.cpu())
            y_pred.append(predictions(output).cpu())
            y_score.append(softmax(output, dim=1)[:, 1].cpu())
    
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_score = torch.cat(y_score).numpy()
    
    accuracy = (y_true == y_pred).mean()
    avg_loss = total_loss / len(loader)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1': f1,
        'auroc': auroc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def visualize_samples_with_gradcam(model, loader, class_names, num_samples=5, 
                                   architecture='resnet18', save_dir='gradcam_visualizations'):
    """Visualize sample predictions with Grad-CAM."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_visualized = 0
    
    for X, y in loader:
        if samples_visualized >= num_samples:
            break
        
        for i in range(X.size(0)):
            if samples_visualized >= num_samples:
                break
            
            # Get single sample
            x_sample = X[i:i+1]
            y_true = y[i].item()
            
            # Get original image (denormalize if needed)
            img = x_sample[0].permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            if img.min() < 0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            elif img.max() > 1:
                img = img / 255.0
            
            # Visualize
            save_path = os.path.join(save_dir, f"sample_{samples_visualized}_true_{class_names[y_true]}.png")
            visualize_gradcam(
                model, x_sample, img, 
                class_names=class_names,
                save_path=save_path,
                architecture=architecture
            )
            
            samples_visualized += 1


def main():
    parser = argparse.ArgumentParser(description='Evaluate fMRI models')
    parser.add_argument('--model', type=str, choices=['cnn', 'mlp'], default='cnn',
                        help='Model type to evaluate')
    parser.add_argument('--architecture', type=str, choices=['resnet18', 'mobilenet'], 
                        default='resnet18', help='CNN architecture (only for CNN model)')
    parser.add_argument('--data_root', type=str, default='../dataset_combined_vertical',
                        help='Root directory containing subject folders')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize with Grad-CAM')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (default: ./checkpoints/fmri_{model}/)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to load (default: best epoch based on validation loss)')
    parser.add_argument('--use_latest', action='store_true',
                        help='Use latest epoch instead of best epoch')
    parser.add_argument('--best_by', type=str, choices=['loss', 'accuracy'], default='loss',
                        help='Criterion for best epoch: loss (lowest val loss) or accuracy (highest val acc)')
    
    args = parser.parse_args()
    
    set_random_seed()
    device = torch.device(args.device)
    
    # Get class names
    class_names = ["face", "house"]
    
    # Load model
    if args.model == 'cnn':
        model = fMRICNN(
            architecture=args.architecture,
            num_classes=2,
            pretrained=True,
            freeze_backbone=False
        )
        checkpoint_dir = args.checkpoint_dir or "./checkpoints/fmri_cnn/"
        
        # Load data
        tr_loader, va_loader, te_loader = get_train_val_test_loaders(
            data_root=args.data_root,
            batch_size=16,
            image_dim=224
        )
    else:  # mlp
        model = fMRIMLP(
            input_dim=2000,
            hidden_dims=[512, 256],
            num_classes=2,
            dropout_rate=0.3
        )
        checkpoint_dir = args.checkpoint_dir or "./checkpoints/fmri_mlp/"
        
        # For MLP, we need to use the voxel extraction
        from train_mlp import get_mlp_loaders
        tr_loader, va_loader, te_loader = get_mlp_loaders(
            data_root=args.data_root,
            batch_size=32,
            top_k=2000,
            image_dim=224
        )
    
    # Restore checkpoint - load best or specified epoch
    print(f"Loading checkpoint from {checkpoint_dir}...")
    try:
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch=*.checkpoint.pth.tar"))
        if checkpoint_files:
            epochs = [int(os.path.basename(f).split("epoch=")[1].split(".")[0]) for f in checkpoint_files]
            
            if args.epoch is not None:
                # Load specified epoch
                if args.epoch not in epochs:
                    print(f"Error: Epoch {args.epoch} not found. Available epochs: {sorted(epochs)}")
                    return
                target_epoch = args.epoch
                print(f"Loading specified epoch: {target_epoch}")
            elif args.use_latest:
                # Load latest epoch
                target_epoch = max(epochs)
                print(f"Loading latest checkpoint: epoch {target_epoch}")
            else:
                # Find best epoch based on validation loss or accuracy
                criterion_name = "validation loss" if args.best_by == 'loss' else "validation accuracy"
                print(f"Finding best epoch based on {criterion_name}...")
                best_epoch = None
                if args.best_by == 'loss':
                    best_metric = float('inf')
                    compare = lambda x, y: x < y
                else:  # accuracy
                    best_metric = float('-inf')
                    compare = lambda x, y: x > y
                
                for ep in epochs:
                    filename = os.path.join(checkpoint_dir, f"epoch={ep}.checkpoint.pth.tar")
                    checkpoint = torch.load(filename, map_location=torch.device(args.device), weights_only=False)
                    if checkpoint.get("stats") and len(checkpoint["stats"]) > 0:
                        # stats format: [val_acc, val_loss, val_auc, train_acc, train_loss, train_auc]
                        if args.best_by == 'loss':
                            metric = checkpoint["stats"][-1][1]  # validation loss
                        else:
                            metric = checkpoint["stats"][-1][0]  # validation accuracy
                        
                        if compare(metric, best_metric):
                            best_metric = metric
                            best_epoch = ep
                
                if best_epoch is None:
                    print("Could not determine best epoch, using latest...")
                    target_epoch = max(epochs)
                else:
                    target_epoch = best_epoch
                    metric_name = "loss" if args.best_by == 'loss' else "accuracy"
                    print(f"Best epoch: {target_epoch} (validation {metric_name}: {best_metric:.4f})")
            
            # Load the selected checkpoint
            filename = os.path.join(checkpoint_dir, f"epoch={target_epoch}.checkpoint.pth.tar")
            checkpoint = torch.load(filename, map_location=torch.device(args.device), weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            epoch = checkpoint["epoch"]
            stats = checkpoint["stats"]
            print(f"Successfully loaded checkpoint from epoch {epoch}")
            
            # Print validation metrics for this epoch
            if stats and len(stats) > 0:
                val_acc = stats[-1][0]
                val_loss = stats[-1][1]
                val_auc = stats[-1][2]
                print(f"  Validation metrics at epoch {epoch}: Acc={val_acc:.4f}, Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        else:
            print("No checkpoint found. Using untrained model.")
            epoch = 0
            stats = []
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Starting evaluation without checkpoint...")
        epoch = 0
        stats = []
    model = model.to(device)
    
    print(f"Loaded model from epoch {epoch}")
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate on all splits
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for split_name, loader in [("Train", tr_loader), ("Validation", va_loader), ("Test", te_loader)]:
        results = evaluate_model(model, loader, criterion, device)
        print(f"\n{split_name} Set:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  ROC-AUC: {results['auroc']:.4f}")
    
    # Confusion matrix on test set
    test_results = evaluate_model(model, te_loader, criterion, device)
    plot_confusion_matrix(
        test_results['y_true'], 
        test_results['y_pred'],
        class_names,
        save_path=f"fmri_{args.model}_confusion_matrix.png"
    )
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*50)
    print(classification_report(
        test_results['y_true'],
        test_results['y_pred'],
        target_names=class_names
    ))
    
    # Grad-CAM visualizations (only for CNN)
    if args.visualize and args.model == 'cnn':
        if not GRADCAM_AVAILABLE:
            print("\nError: Grad-CAM visualization requires opencv-python.")
            print("Install with: pip install opencv-python")
        else:
            print("\nGenerating Grad-CAM visualizations...")
            visualize_samples_with_gradcam(
                model, te_loader, class_names,
                num_samples=args.num_samples,
                architecture=args.architecture
            )


if __name__ == "__main__":
    main()


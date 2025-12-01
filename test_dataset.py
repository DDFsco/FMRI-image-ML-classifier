"""
Test script to verify dataset loading
Usage: python test_dataset.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_train_val_test_datasets, fMRIDataset

def main():
    """Test dataset loading."""
    data_root = "../dataset_combined_vertical"
    
    print("Testing dataset loading...")
    print(f"Data root: {data_root}")
    
    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"ERROR: Data root not found: {data_root}")
        print("Please ensure dataset_combined_vertical is in the project2 directory")
        return
    
    # Load datasets
    try:
        tr, va, te, standardizer = get_train_val_test_datasets(
            data_root=data_root,
            image_dim=224
        )
        
        print(f"\n✓ Successfully loaded datasets!")
        print(f"  Train: {len(tr)} images")
        print(f"  Val: {len(va)} images")
        print(f"  Test: {len(te)} images")
        print(f"  Total: {len(tr) + len(va) + len(te)} images")
        
        # Test loading a sample
        print("\nTesting sample loading...")
        img, label = tr[0]
        print(f"  Sample image shape: {img.shape}")
        print(f"  Sample label: {label.item()} ({tr.get_class_name(label.item())})")
        print(f"  Image value range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Check class distribution
        print("\nClass distribution:")
        train_labels = [tr[i][1].item() for i in range(len(tr))]
        face_count = sum(1 for l in train_labels if l == 0)
        house_count = sum(1 for l in train_labels if l == 1)
        print(f"  Face (0): {face_count} images")
        print(f"  House (1): {house_count} images")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


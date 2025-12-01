"""
EECS 442 Final Project - BrainMapNet

fMRI Dataset
    Loads fMRI brain activation maps from subject folders
    Structure: dataset_combined_vertical/{subject}/combined_vertical/brain_cond{1-12}_vertical.png
    Labels: cond1-6 = house (1), cond7-12 = face (0)
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
from imageio.v2 import imread
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
import os

# Try to import from parent directory (for local use)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import config, set_random_seed
    _HAS_UTILS = True
except (ImportError, NameError):
    # For notebook/Colab use, define set_random_seed here
    _HAS_UTILS = False
    SEED = 445  # Default seed
    
    def set_random_seed():
        """Set random seed for reproducibility."""
        import random
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # config function not needed for notebook
    def config(attr):
        raise NotImplementedError("config() not available in notebook mode. Use direct configuration.")

# Import ImageStandardizer from train_common
try:
    from train_common import ImageStandardizer
except ImportError:
    # Fallback: define ImageStandardizer here if needed
    import numpy as np
    import numpy.typing as npt
    
    class ImageStandardizer:
        """Standardize a batch of images to mean 0 and variance 1."""
        def __init__(self) -> None:
            self.image_mean = None
            self.image_std = None
        
        def fit(self, X: npt.NDArray) -> None:
            """Calculate per-channel mean and standard deviation from dataset X."""
            self.image_mean = np.mean(X, axis=(0, 1, 2))
            self.image_std = np.std(X, axis=(0, 1, 2))
        
        def transform(self, X: npt.NDArray) -> npt.NDArray:
            """Return standardized dataset given dataset X."""
            standardized = (X - self.image_mean) / (self.image_std + 1e-8)
            return standardized


__all__ = [
    "get_train_val_test_loaders", 
    "get_train_val_test_datasets", 
    "resize", 
    "fMRIDataset"
]


def get_train_val_test_loaders(data_root: str, batch_size: int, image_dim: int = 224, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return DataLoaders for train, val and test splits."""
    tr, va, te, _ = get_train_val_test_datasets(
        data_root=data_root,
        image_dim=image_dim,
        **kwargs
    )

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader


class fMRIDataset(Dataset):
    """Dataset class for fMRI brain activation maps from subject folders."""

    def __init__(
        self, 
        data_root: str,
        subjects: list,
        partition: str = "train",
        image_dim: int = 224,
        standardizer: Optional[ImageStandardizer] = None,
        use_imagenet_norm: bool = True
    ) -> None:
        """Initialize fMRI dataset.
        
        Args:
            data_root: Root directory containing subject folders (e.g., dataset_combined_vertical)
            subjects: List of subject IDs to include in this dataset
            partition: Partition name (for logging)
            image_dim: Target image dimension for resizing
            standardizer: ImageStandardizer instance (optional)
            use_imagenet_norm: If True, use ImageNet normalization (for pretrained models)
        """
        super().__init__()
        
        self.data_root = data_root
        self.subjects = subjects
        self.partition = partition
        self.image_dim = image_dim
        self.standardizer = standardizer
        self.use_imagenet_norm = use_imagenet_norm
        
        # ImageNet normalization parameters (for pretrained models)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        
        # Load all images from specified subjects
        self.image_paths = []
        self.labels = []
        
        print(f"Loading {partition} data from {len(subjects)} subjects...")
        for subject in subjects:
            subject_dir = os.path.join(data_root, subject, "combined_vertical")
            if not os.path.exists(subject_dir):
                print(f"Warning: Subject directory not found: {subject_dir}")
                continue
            
            # Load cond1-12 images
            for cond in range(1, 13):
                image_name = f"brain_cond{cond}_vertical.png"
                image_path = os.path.join(subject_dir, image_name)
                
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    # cond1-6 = house (1), cond7-12 = face (0)
                    label = 1 if cond <= 6 else 0
                    self.labels.append(label)
                else:
                    print(f"Warning: Image not found: {image_path}")
        
        print(f"Loaded {len(self.image_paths)} images for {partition} set")
        
        # Class names
        self.class_names = ["face", "house"]
        self.label_to_name = {0: "face", 1: "house"}

    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, label) pair at index `idx` of dataset."""
        # Load image
        image_path = self.image_paths[idx]
        image = imread(image_path)
        
        # Handle grayscale images - convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Resize if needed
        if self.image_dim and self.image_dim > 0:
            image = resize_image(image, self.image_dim)
        
        # Convert to tensor and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to (C, H, W) format
        image = image.transpose(2, 0, 1)
        
        # Apply normalization
        if self.use_imagenet_norm:
            # Use ImageNet normalization for pretrained models
            image = (image - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]
        elif self.standardizer is not None:
            # Use dataset-specific standardization
            # Reshape to (H, W, C) for standardizer
            image_hwc = image.transpose(1, 2, 0)
            image_hwc = self.standardizer.transform(image_hwc[None, ...])[0]
            # Convert back to (C, H, W)
            image = image_hwc.transpose(2, 0, 1)
        
        label = self.labels[idx]
        
        return torch.from_numpy(image).float(), torch.tensor(label).long()
    
    def get_class_name(self, label: int) -> str:
        """Return class name for a label."""
        return self.label_to_name.get(label, f"unknown_{label}")


def resize_image(image: np.ndarray, image_dim: int) -> np.ndarray:
    """Resize image to specified dimension.
    
    Args:
        image: Input image as numpy array
        image_dim: Target image dimension (square)
        
    Returns:
        Resized image as numpy array
    """
    image_size = (image_dim, image_dim)
    img_pil = Image.fromarray(image.astype(np.uint8))
    img_resized = img_pil.resize(image_size, resample=Image.Resampling.BICUBIC)
    return np.asarray(img_resized)


def get_train_val_test_datasets(
    data_root: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 445,
    image_dim: int = 224
) -> tuple[fMRIDataset, fMRIDataset, fMRIDataset, ImageStandardizer]:
    """Return fMRIDatasets and image standardizer.
    
    Performs subject-level split (not image-level) to avoid data leakage.
    Each subject's images stay in the same partition.
    
    Args:
        data_root: Root directory containing subject folders
        test_size: Proportion of subjects for test set
        val_size: Proportion of subjects for validation set
        random_state: Random seed for splitting
        image_dim: Target image dimension
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, standardizer)
    """
    set_random_seed()
    
    # Get all subject folders
    all_subjects = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))
                   and os.path.exists(os.path.join(data_root, d, "combined_vertical"))]
    all_subjects.sort()
    
    print(f"Found {len(all_subjects)} subjects in {data_root}")
    
    # Split subjects into train/val/test
    # First split: train+val vs test
    train_val_subjects, test_subjects = train_test_split(
        all_subjects,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    # Create standardizer (for potential use, but we'll use ImageNet norm by default)
    # Create standardizer and fit on training data
    print("Computing image statistics from training set...")
    # Sample some images to compute statistics
    sample_dataset = fMRIDataset(data_root, train_subjects[:10], partition="train", image_dim=image_dim, use_imagenet_norm=False)
    train_images = []
    for i in range(min(100, len(sample_dataset))):
        img, _ = sample_dataset[i]
        train_images.append(img.numpy())
    train_images = np.array(train_images)
    
    # Reshape to (N, H, W, C) for standardizer
    train_images = train_images.transpose(0, 2, 3, 1)
    
    standardizer = ImageStandardizer()
    standardizer.fit(train_images)
    
    # Create datasets with ImageNet normalization (for pretrained models)
    # Set use_imagenet_norm=True to use ImageNet stats, False to use dataset stats
    tr = fMRIDataset(data_root, train_subjects, partition="train", image_dim=image_dim, 
                     standardizer=standardizer, use_imagenet_norm=True)
    va = fMRIDataset(data_root, val_subjects, partition="val", image_dim=image_dim,
                     standardizer=standardizer, use_imagenet_norm=True)
    te = fMRIDataset(data_root, test_subjects, partition="test", image_dim=image_dim,
                     standardizer=standardizer, use_imagenet_norm=True)
    
    return tr, va, te, standardizer


if __name__ == "__main__":
    # Test dataset loading
    data_root = "../dataset_combined_vertical"
    tr, va, te, standardizer = get_train_val_test_datasets(data_root)
    print(f"\nTrain: {len(tr)} images")
    print(f"Val: {len(va)} images")
    print(f"Test: {len(te)} images")
    print(f"Mean: {standardizer.image_mean}")
    print(f"Std: {standardizer.image_std}")


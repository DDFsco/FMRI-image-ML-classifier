"""
EECS 442 Final Project - BrainMapNet

Grad-CAM Visualization
    Implementation of Grad-CAM for visualizing important regions in fMRI activation maps
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python (cv2) not installed. Grad-CAM will not work.")
    print("Install with: pip install opencv-python")

__all__ = ["GradCAM", "visualize_gradcam"]


class GradCAM:
    """Grad-CAM implementation for CNN models."""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize Grad-CAM.
        
        Args:
            model: The trained model
            target_layer: The target convolutional layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save activation maps."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate Class Activation Map (CAM).
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            class_idx: Class index to visualize. If None, uses predicted class.
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Shape: (C, H, W)
        activations = self.activations[0]  # Shape: (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def visualize_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: list = None,
    save_path: str = None,
    architecture: str = "resnet18"
) -> None:
    """Visualize Grad-CAM heatmap overlaid on original image.
    
    Args:
        model: Trained model
        input_tensor: Input tensor of shape (1, C, H, W)
        original_image: Original image as numpy array (H, W, C)
        class_names: List of class names (e.g., ["face", "house"])
        save_path: Path to save the visualization
        architecture: Model architecture ("resnet18" or "mobilenet")
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python (cv2) is required for Grad-CAM visualization. Install with: pip install opencv-python")
    
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = F.softmax(output, dim=1)[0, pred_class].item()
    
    # Initialize Grad-CAM
    if architecture == "resnet18":
        if hasattr(model, 'backbone'):
            target_layer = model.backbone.layer4
        else:
            target_layer = model.layer4
    else:  # mobilenet
        if hasattr(model, 'backbone'):
            features = model.backbone.features
            target_layer = features[-1]
        else:
            target_layer = model.features[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor, pred_class)
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Convert original image to RGB if needed
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image, original_image, original_image], axis=-1)
    elif original_image.shape[2] == 1:
        original_image = np.repeat(original_image, 3, axis=2)
    
    # Normalize original image to [0, 1]
    if original_image.max() > 1:
        original_image = original_image.astype(np.float32) / 255.0
    
    # Overlay heatmap on original image
    heatmap = heatmap.astype(np.float32) / 255.0
    overlayed = 0.6 * original_image + 0.4 * heatmap
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original fMRI Map")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlayed)
    class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
    axes[2].set_title(f"Overlay\nPredicted: {class_name} ({pred_prob:.3f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.close()



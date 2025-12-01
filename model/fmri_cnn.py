"""
EECS 442 Final Project - BrainMapNet

fMRI CNN Classifier
    Uses pretrained ResNet-18 or MobileNet for classifying fMRI activation maps
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights

__all__ = ["fMRICNN"]


class fMRICNN(nn.Module):
    """CNN classifier using pretrained ResNet-18 or MobileNet for fMRI classification."""
    
    def __init__(
        self, 
        architecture: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ) -> None:
        """Initialize the fMRI CNN classifier.
        
        Args:
            architecture: Either "resnet18" or "mobilenet"
            num_classes: Number of output classes (default: 2 for face/house)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone (for fine-tuning)
        """
        super().__init__()
        
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        
        if self.architecture == "resnet18":
            if pretrained:
                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            
            # Replace the final fully connected layer
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif self.architecture == "mobilenet":
            if pretrained:
                self.backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.mobilenet_v3_small(weights=None)
            
            # Replace the classifier
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'resnet18' or 'mobilenet'")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the final layer
            if self.architecture == "resnet18":
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            else:  # mobilenet
                for param in self.backbone.classifier[-1].parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer.
        
        Useful for Grad-CAM visualization.
        """
        if self.architecture == "resnet18":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:  # mobilenet
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x


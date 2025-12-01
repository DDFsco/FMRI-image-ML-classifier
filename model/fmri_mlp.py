"""
EECS 442 Final Project - BrainMapNet

fMRI MLP Baseline
    Simple MLP classifier using top activated voxels as input
"""

import torch
import torch.nn as nn

__all__ = ["fMRIMLP"]


class fMRIMLP(nn.Module):
    """MLP baseline classifier for fMRI data using top activated voxels."""
    
    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: list = [512, 256],
        num_classes: int = 2,
        dropout_rate: float = 0.3
    ) -> None:
        """Initialize the fMRI MLP classifier.
        
        Args:
            input_dim: Dimension of input vector (number of top voxels, default: 2000)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            num_classes: Number of output classes (default: 2 for face/house)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)


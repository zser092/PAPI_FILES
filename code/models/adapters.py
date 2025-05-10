"""
Adapter modules for the PaPI framework.

This module implements lightweight adapter layers that can be inserted into transformer models
to enable efficient fine-tuning with minimal parameter updates (1-3% of the model parameters).
"""

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    Adapter module as described in the PaPI paper.
    
    Adapters are small, parameter-efficient modules inserted into transformer models that:
    - Enable efficient transfer of pre-trained models to new tasks
    - Minimize retraining of base model parameters
    - Are typically implemented as feed-forward layers that project features to a smaller dimension and back
    
    Args:
        input_dim (int): Dimension of the input features
        bottleneck_dim (int): Dimension of the bottleneck (reduced dimension)
        adapter_non_linearity (str): Non-linearity to use in the adapter ('relu' or 'gelu')
        adapter_initializer_range (float): Range for weight initialization
    """
    def __init__(
        self,
        input_dim,
        bottleneck_dim,
        adapter_non_linearity="relu",
        adapter_initializer_range=0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        
        # Non-linearity
        if adapter_non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif adapter_non_linearity == "gelu":
            self.non_linearity = nn.GELU()
        else:
            raise ValueError(f"Unknown non-linearity: {adapter_non_linearity}")
        
        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # Initialize weights
        self._init_weights(adapter_initializer_range)
    
    def _init_weights(self, adapter_initializer_range):
        """Initialize the weights of the adapter layers."""
        nn.init.normal_(self.down_proj.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass through the adapter.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, input_dim]
        """
        # Down projection
        down_projected = self.down_proj(hidden_states)
        
        # Apply non-linearity
        activated = self.non_linearity(down_projected)
        
        # Up projection
        up_projected = self.up_proj(activated)
        
        # Residual connection
        return hidden_states + up_projected


class AdapterLayer(nn.Module):
    """
    Adapter layer that can be inserted into a transformer layer.
    
    This module wraps the adapter with layer normalization as used in the PaPI framework.
    
    Args:
        input_dim (int): Dimension of the input features
        bottleneck_dim (int): Dimension of the bottleneck (reduced dimension)
        adapter_non_linearity (str): Non-linearity to use in the adapter ('relu' or 'gelu')
        adapter_initializer_range (float): Range for weight initialization
        layer_norm_eps (float): Epsilon for layer normalization
    """
    def __init__(
        self,
        input_dim,
        bottleneck_dim,
        adapter_non_linearity="relu",
        adapter_initializer_range=0.01,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.adapter = Adapter(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_non_linearity=adapter_non_linearity,
            adapter_initializer_range=adapter_initializer_range,
        )
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
    
    def forward(self, hidden_states):
        """
        Forward pass through the adapter layer.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, input_dim]
        """
        # Apply adapter
        adapter_output = self.adapter(hidden_states)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(adapter_output)
        
        return normalized_output


class TaskSpecificAdapterBlock(nn.Module):
    """
    Task-specific adapter block that contains multiple adapter layers.
    
    This module represents a task-specific pathway in the PaPI framework.
    
    Args:
        input_dim (int): Dimension of the input features
        bottleneck_dim (int): Dimension of the bottleneck (reduced dimension)
        num_layers (int): Number of adapter layers
        adapter_non_linearity (str): Non-linearity to use in the adapter ('relu' or 'gelu')
        adapter_initializer_range (float): Range for weight initialization
        layer_norm_eps (float): Epsilon for layer normalization
    """
    def __init__(
        self,
        input_dim,
        bottleneck_dim,
        num_layers=2,
        adapter_non_linearity="relu",
        adapter_initializer_range=0.01,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AdapterLayer(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                adapter_non_linearity=adapter_non_linearity,
                adapter_initializer_range=adapter_initializer_range,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states):
        """
        Forward pass through the task-specific adapter block.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, input_dim]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
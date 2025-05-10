"""
Routing classifier for the PaPI framework.

This module implements the softmax-based routing mechanism that dynamically
allocates input samples to appropriate tasks, enhancing computational efficiency
and adaptability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingClassifier(nn.Module):
    """
    Softmax-based routing classifier as described in the PaPI paper.
    
    The routing classifier dynamically allocates input samples to appropriate tasks
    based on the features extracted from the base model.
    
    Args:
        hidden_dim (int): Dimension of the input features
        num_tasks (int): Number of tasks to route between
        temperature (float): Temperature parameter for softmax (controls sharpness)
    """
    def __init__(self, hidden_dim, num_tasks, temperature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.temperature = temperature
        
        # Task-specific weight vectors for routing
        self.task_weights = nn.Parameter(torch.Tensor(num_tasks, hidden_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the task weight vectors."""
        nn.init.normal_(self.task_weights, std=0.01)
    
    def forward(self, hidden_states):
        """
        Forward pass through the routing classifier.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_dim]
                                         or [batch_size, hidden_dim]
            
        Returns:
            tuple:
                - task_probs (torch.Tensor): Task probabilities of shape [batch_size, num_tasks]
                - task_indices (torch.Tensor): Selected task indices of shape [batch_size]
        """
        # If input has sequence dimension, use CLS token or mean pooling
        if len(hidden_states.shape) == 3:
            # Use CLS token (first token) or mean pooling
            # In the paper, they use the CLS token for classification tasks
            hidden_states = hidden_states[:, 0]  # Use CLS token
        
        # Compute routing logits: w_t^T * h(x)
        # Shape: [batch_size, num_tasks]
        routing_logits = torch.matmul(hidden_states, self.task_weights.transpose(0, 1))
        
        # Apply temperature scaling
        scaled_logits = routing_logits / self.temperature
        
        # Compute softmax probabilities
        # P(t|x) = exp(w_t^T * h(x) / temperature) / sum_k(exp(w_k^T * h(x) / temperature))
        task_probs = F.softmax(scaled_logits, dim=-1)
        
        # Get the most probable task
        task_indices = torch.argmax(task_probs, dim=-1)
        
        return task_probs, task_indices


class RoutingMechanism(nn.Module):
    """
    Complete routing mechanism that combines feature extraction and routing.
    
    Args:
        hidden_dim (int): Dimension of the input features
        num_tasks (int): Number of tasks to route between
        temperature (float): Temperature parameter for softmax
        feature_extractor (nn.Module, optional): Feature extractor module
    """
    def __init__(self, hidden_dim, num_tasks, temperature=1.0, feature_extractor=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = RoutingClassifier(hidden_dim, num_tasks, temperature)
    
    def forward(self, inputs, extract_features=True):
        """
        Forward pass through the routing mechanism.
        
        Args:
            inputs: Input data
            extract_features (bool): Whether to extract features or use inputs directly
            
        Returns:
            tuple:
                - task_probs (torch.Tensor): Task probabilities
                - task_indices (torch.Tensor): Selected task indices
                - features (torch.Tensor): Extracted features (if feature_extractor is provided)
        """
        if extract_features and self.feature_extractor is not None:
            features = self.feature_extractor(inputs)
        else:
            features = inputs
        
        task_probs, task_indices = self.classifier(features)
        
        if extract_features and self.feature_extractor is not None:
            return task_probs, task_indices, features
        else:
            return task_probs, task_indices
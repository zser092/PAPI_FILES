"""
Elastic Weight Consolidation (EWC) regularization for the PaPI framework.

This module implements the EWC regularization technique that prevents catastrophic
forgetting by penalizing changes to parameters that are important for previously
learned tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWCRegularization:
    """
    Elastic Weight Consolidation (EWC) regularization as described in the PaPI paper.
    
    EWC is a regularization technique used to prevent catastrophic forgetting by:
    1. Identifying and protecting crucial network parameters that are important for previously learned tasks
    2. "Elastically" anchoring important weights to prevent complete overwriting
    3. Using the Fisher Information Matrix to quantify parameter importance
    4. Adding a regularization term to the loss function that penalizes changes to important parameters
    
    Args:
        model (nn.Module): The model to apply EWC to
        dataset (torch.utils.data.Dataset): Dataset of samples from previous tasks
        device (torch.device): Device to use for computation
        fisher_sample_size (int): Number of samples to use for Fisher Information Matrix computation
        fisher_alpha (float): Scaling factor for the Fisher Information Matrix
    """
    def __init__(self, model, dataset, device, fisher_sample_size=500, fisher_alpha=1000.0):
        self.model = model
        self.device = device
        self.fisher_sample_size = min(fisher_sample_size, len(dataset))
        self.fisher_alpha = fisher_alpha
        
        # Get a subset of the dataset for Fisher computation
        indices = torch.randperm(len(dataset))[:self.fisher_sample_size]
        self.fisher_dataset = torch.utils.data.Subset(dataset, indices)
        
        # Initialize Fisher Information Matrix and parameter means
        self.fisher_matrix = {}
        self.parameter_means = {}
        
        # Compute Fisher Information Matrix and parameter means
        self._compute_fisher_and_means()
    
    def _compute_fisher_and_means(self):
        """
        Compute the Fisher Information Matrix and parameter means.
        
        The Fisher Information Matrix is computed as:
        F_i = E_x~D[(∂log(P(y|x, θ))/∂θ_i)²]
        
        where:
        - F_i is the Fisher Information for parameter θ_i
        - D is the data distribution
        - P(y|x, θ) is the model's output probability
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Fisher matrix with zeros for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param.data)
                self.parameter_means[name] = param.data.clone()
        
        # Create a dataloader for the Fisher dataset
        fisher_loader = torch.utils.data.DataLoader(
            self.fisher_dataset, batch_size=32, shuffle=True
        )
        
        # Compute Fisher Information Matrix
        for batch in fisher_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Compute log probabilities
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Select log probabilities for the target classes
            target_log_probs = log_probs.gather(1, labels.unsqueeze(1))
            
            # Compute gradients
            for i in range(target_log_probs.size(0)):
                self.model.zero_grad()
                target_log_probs[i].backward(retain_graph=(i < target_log_probs.size(0) - 1))
                
                # Accumulate squared gradients in the Fisher matrix
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_matrix[name] += param.grad.data ** 2 / self.fisher_sample_size
        
        # Scale the Fisher matrix by the alpha parameter
        for name in self.fisher_matrix:
            self.fisher_matrix[name] *= self.fisher_alpha
    
    def penalty(self):
        """
        Compute the EWC penalty term.
        
        The EWC penalty is computed as:
        L_reg = (λ/2) * sum_i F_i * (θ_i - θ*_i)²
        
        where:
        - λ is the regularization strength (fisher_alpha)
        - F_i is the Fisher Information for parameter θ_i
        - θ_i is the current value of parameter i
        - θ*_i is the optimal value of parameter i for previous tasks
        
        Returns:
            torch.Tensor: The EWC penalty term
        """
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix and param.requires_grad:
                # Compute squared distance between current and optimal parameters
                penalty += (self.fisher_matrix[name] * (param.data - self.parameter_means[name]) ** 2).sum()
        
        return penalty / 2.0
    
    def update_fisher_and_means(self, new_dataset):
        """
        Update the Fisher Information Matrix and parameter means with data from a new task.
        
        Args:
            new_dataset (torch.utils.data.Dataset): Dataset of samples from the new task
        """
        # Store old values
        old_fisher = {name: self.fisher_matrix[name].clone() for name in self.fisher_matrix}
        old_means = {name: self.parameter_means[name].clone() for name in self.parameter_means}
        
        # Update dataset
        indices = torch.randperm(len(new_dataset))[:self.fisher_sample_size]
        self.fisher_dataset = torch.utils.data.Subset(new_dataset, indices)
        
        # Compute new Fisher and means
        self._compute_fisher_and_means()
        
        # Merge old and new Fisher matrices and means
        for name in self.fisher_matrix:
            # Average the Fisher matrices
            self.fisher_matrix[name] = (old_fisher[name] + self.fisher_matrix[name]) / 2
            
            # Keep the old means (they represent the optimal parameters for previous tasks)
            self.parameter_means[name] = old_means[name]


class EWCLoss(nn.Module):
    """
    Loss function that combines task-specific loss with EWC regularization.
    
    Args:
        task_loss_fn (callable): Task-specific loss function
        ewc (EWCRegularization): EWC regularization object
        lambda_ewc (float): Weight for the EWC regularization term
    """
    def __init__(self, task_loss_fn, ewc, lambda_ewc=1.0):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.ewc = ewc
        self.lambda_ewc = lambda_ewc
    
    def forward(self, outputs, targets):
        """
        Compute the combined loss.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            torch.Tensor: Combined loss (task loss + EWC regularization)
        """
        # Compute task-specific loss
        task_loss = self.task_loss_fn(outputs, targets)
        
        # Compute EWC regularization term
        ewc_loss = self.ewc.penalty()
        
        # Combine losses
        total_loss = task_loss + self.lambda_ewc * ewc_loss
        
        return total_loss
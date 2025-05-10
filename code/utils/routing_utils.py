"""
Utility functions for the routing mechanism in the PaPI framework.

This module provides utility functions for training and evaluating the routing mechanism,
as well as for tracking energy consumption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def train_routing_classifier(model, dataloader, optimizer, device, num_epochs=3):
    """
    Train the routing classifier.
    
    Args:
        model (nn.Module): The PaPI model with routing mechanism
        dataloader (torch.utils.data.DataLoader): DataLoader with (inputs, task_ids) pairs
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to use for training
        num_epochs (int): Number of training epochs
        
    Returns:
        dict: Training statistics
    """
    model.train()
    
    # Statistics
    stats = {
        "epoch_losses": [],
        "epoch_accuracies": [],
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Unpack batch
            inputs, task_ids = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            task_ids = task_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**inputs)
            routing_probs = outputs["routing_probs"]
            
            # Compute loss
            loss = F.cross_entropy(routing_probs, task_ids)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            predicted_task_ids = torch.argmax(routing_probs, dim=1)
            correct_predictions += (predicted_task_ids == task_ids).sum().item()
            total_predictions += task_ids.size(0)
        
        # Compute epoch statistics
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        
        # Store statistics
        stats["epoch_losses"].append(avg_epoch_loss)
        stats["epoch_accuracies"].append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
    
    return stats


def evaluate_routing_accuracy(model, dataloader, device):
    """
    Evaluate the routing accuracy.
    
    Args:
        model (nn.Module): The PaPI model with routing mechanism
        dataloader (torch.utils.data.DataLoader): DataLoader with (inputs, task_ids) pairs
        device (torch.device): Device to use for evaluation
        
    Returns:
        float: Routing accuracy
    """
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating routing accuracy"):
            # Unpack batch
            inputs, task_ids = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            task_ids = task_ids.to(device)
            
            # Forward pass
            outputs = model(**inputs)
            routing_probs = outputs["routing_probs"]
            
            # Compute accuracy
            predicted_task_ids = torch.argmax(routing_probs, dim=1)
            correct_predictions += (predicted_task_ids == task_ids).sum().item()
            total_predictions += task_ids.size(0)
    
    routing_accuracy = correct_predictions / total_predictions
    print(f"Routing Accuracy: {routing_accuracy:.4f}")
    
    return routing_accuracy


def track_energy_consumption(model, batch_size=1, seq_length=128, num_iterations=1):
    """
    Track the energy consumption of the model.
    
    Args:
        model (nn.Module): The PaPI model
        batch_size (int): Batch size
        seq_length (int): Sequence length
        num_iterations (int): Number of iterations
        
    Returns:
        dict: Energy consumption statistics
    """
    # Get energy consumption estimates
    energy_stats = model.get_energy_consumption(batch_size, seq_length)
    
    # Scale by number of iterations
    energy_stats["total_energy"] = energy_stats["estimated_energy"] * num_iterations
    
    # Estimate CO2 emissions (assuming 400 gCO2/kWh as mentioned in the paper)
    # Convert energy from arbitrary units to kWh (this is a rough approximation)
    # The paper mentions that PaPI uses 2% of the energy of full fine-tuning
    # Full fine-tuning: 4.608 kWh, PaPI: 0.09216 kWh
    energy_kwh = energy_stats["total_energy"] * 0.09216 / energy_stats["estimated_energy"]
    co2_emissions = energy_kwh * 400  # 400 gCO2/kWh
    
    energy_stats["energy_kwh"] = energy_kwh
    energy_stats["co2_emissions_g"] = co2_emissions
    
    return energy_stats


def compute_forgetting_rate(model, task_id, initial_accuracy, current_accuracy):
    """
    Compute the forgetting rate for a specific task.
    
    The forgetting rate is defined as the relative decrease in accuracy on a task
    after training on subsequent tasks.
    
    Args:
        model (nn.Module): The PaPI model
        task_id (int): Task ID
        initial_accuracy (float): Initial accuracy on the task
        current_accuracy (float): Current accuracy on the task
        
    Returns:
        float: Forgetting rate
    """
    if initial_accuracy == 0:
        return 0.0
    
    forgetting_rate = (initial_accuracy - current_accuracy) / initial_accuracy
    return max(0.0, forgetting_rate)  # Ensure non-negative


def compute_backward_transfer(accuracies):
    """
    Compute the backward transfer.
    
    Backward transfer measures how learning new tasks affects the performance on
    previously learned tasks. Negative values indicate catastrophic forgetting.
    
    Args:
        accuracies (list): List of lists, where accuracies[i][j] is the accuracy
                          on task i after training on task j
        
    Returns:
        float: Backward transfer
    """
    n_tasks = len(accuracies)
    backward_transfer = 0.0
    count = 0
    
    for i in range(n_tasks - 1):  # For each task except the last one
        for j in range(i + 1, n_tasks):  # For each subsequent task
            # Accuracy on task i after training on task j, minus
            # accuracy on task i after training on task i
            backward_transfer += accuracies[i][j] - accuracies[i][i]
            count += 1
    
    return backward_transfer / count if count > 0 else 0.0


def compute_forward_transfer(accuracies):
    """
    Compute the forward transfer.
    
    Forward transfer measures how learning previous tasks affects the performance on
    new tasks. Positive values indicate positive transfer.
    
    Args:
        accuracies (list): List of lists, where accuracies[i][j] is the accuracy
                          on task i after training on task j
        
    Returns:
        float: Forward transfer
    """
    n_tasks = len(accuracies)
    forward_transfer = 0.0
    count = 0
    
    for i in range(1, n_tasks):  # For each task except the first one
        # Accuracy on task i before training on it (after training on task i-1)
        # minus random initialization accuracy
        forward_transfer += accuracies[i][i-1] - accuracies[i][0]
        count += 1
    
    return forward_transfer / count if count > 0 else 0.0
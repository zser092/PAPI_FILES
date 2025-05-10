"""
PaPI (Pathway-based Progressive Inference) model for energy-efficient continual learning.

This module implements the PaPI model architecture as described in the papers, combining
RoBERTa-base with lightweight adapters, Elastic Weight Consolidation (EWC), and an
optional softmax-based routing mechanism.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from .adapters import TaskSpecificAdapterBlock
from .routing_classifier import RoutingMechanism


class PaPIModel(nn.Module):
    """
    PaPI (Pathway-based Progressive Inference) model for energy-efficient continual learning.
    
    The PaPI model architecture is built upon RoBERTa-base, augmented with:
    - Task-specific adapters for parameter-efficient fine-tuning
    - Optional routing mechanism for dynamic task selection
    
    Args:
        num_tasks (int): Number of tasks
        num_labels_per_task (list): List of number of labels for each task
        pretrained_model_name (str): Name of the pretrained model to use
        adapter_bottleneck_dim (int): Dimension of the adapter bottleneck
        use_routing (bool): Whether to use the routing mechanism
        routing_temperature (float): Temperature parameter for the routing mechanism
        freeze_base_model (bool): Whether to freeze the base model parameters
    """
    def __init__(
        self,
        num_tasks,
        num_labels_per_task,
        pretrained_model_name="roberta-base",
        adapter_bottleneck_dim=64,
        use_routing=True,
        routing_temperature=1.0,
        freeze_base_model=True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_labels_per_task = num_labels_per_task
        self.use_routing = use_routing
        
        # Load the base model (RoBERTa)
        self.base_model = RobertaModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = self.base_model.config.hidden_size
        
        # Freeze base model parameters if specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Create task-specific adapters
        self.adapters = nn.ModuleList([
            TaskSpecificAdapterBlock(
                input_dim=self.hidden_dim,
                bottleneck_dim=adapter_bottleneck_dim,
            )
            for _ in range(num_tasks)
        ])
        
        # Create task-specific classification heads
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_dim, num_labels)
            for num_labels in num_labels_per_task
        ])
        
        # Create routing mechanism if specified
        if use_routing:
            self.routing = RoutingMechanism(
                hidden_dim=self.hidden_dim,
                num_tasks=num_tasks,
                temperature=routing_temperature,
            )
        else:
            self.routing = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classification heads."""
        for classifier in self.classifiers:
            nn.init.normal_(classifier.weight, std=0.02)
            nn.init.zeros_(classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_id=None):
        """
        Forward pass through the PaPI model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            token_type_ids (torch.Tensor, optional): Token type IDs
            task_id (int, optional): Task ID. If None and use_routing is True, the task will be
                                    determined by the routing mechanism.
            
        Returns:
            dict: Dictionary containing:
                - logits (torch.Tensor): Classification logits
                - task_id (int): Task ID used (provided or determined by routing)
                - routing_probs (torch.Tensor, optional): Routing probabilities if routing is used
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = base_outputs.last_hidden_state
        
        # Determine task ID using routing if not provided
        routing_probs = None
        if task_id is None and self.use_routing:
            routing_probs, task_id = self.routing(hidden_states)
        
        # If task_id is still None, default to the first task
        if task_id is None:
            task_id = 0
        
        # Apply task-specific adapter
        if isinstance(task_id, torch.Tensor):
            # Handle batch with different task IDs
            adapted_outputs = torch.zeros_like(hidden_states)
            for i, t_id in enumerate(task_id):
                t_id = t_id.item()
                adapted_outputs[i] = self.adapters[t_id](hidden_states[i].unsqueeze(0)).squeeze(0)
            
            # Apply task-specific classifiers
            logits = []
            for i, t_id in enumerate(task_id):
                t_id = t_id.item()
                cls_output = adapted_outputs[i, 0]  # Use CLS token
                logits.append(self.classifiers[t_id](cls_output))
            logits = torch.stack(logits)
        else:
            # Apply the same task adapter to the whole batch
            adapted_outputs = self.adapters[task_id](hidden_states)
            
            # Apply task-specific classifier
            cls_output = adapted_outputs[:, 0]  # Use CLS token
            logits = self.classifiers[task_id](cls_output)
        
        return {
            "logits": logits,
            "task_id": task_id,
            "routing_probs": routing_probs,
        }
    
    def get_energy_consumption(self, batch_size=1, seq_length=128):
        """
        Estimate the energy consumption of the model.
        
        The energy consumption is estimated using the formula:
        E = α * |W_updated| * N
        
        where:
        - α is the energy cost per weight update
        - |W_updated| is the number of trainable parameters
        - N is the number of training iterations
        
        Args:
            batch_size (int): Batch size
            seq_length (int): Sequence length
            
        Returns:
            dict: Dictionary containing energy consumption estimates
        """
        # Count trainable parameters
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count total parameters
        num_total_params = sum(p.numel() for p in self.parameters())
        
        # Estimate FLOPs for a forward pass
        # This is a rough estimate based on the model architecture
        hidden_dim = self.hidden_dim
        adapter_bottleneck_dim = self.adapters[0].layers[0].adapter.bottleneck_dim
        
        # FLOPs for base model (if not frozen)
        base_model_flops = 0
        if any(p.requires_grad for p in self.base_model.parameters()):
            # Rough estimate for transformer model
            base_model_flops = 12 * batch_size * seq_length * hidden_dim * hidden_dim
        
        # FLOPs for adapters
        # Down projection: batch_size * seq_length * hidden_dim * adapter_bottleneck_dim
        # Non-linearity: batch_size * seq_length * adapter_bottleneck_dim
        # Up projection: batch_size * seq_length * adapter_bottleneck_dim * hidden_dim
        adapter_flops = batch_size * seq_length * (
            2 * hidden_dim * adapter_bottleneck_dim + adapter_bottleneck_dim
        )
        
        # FLOPs for classifier
        classifier_flops = batch_size * hidden_dim * max(self.num_labels_per_task)
        
        # Total FLOPs
        total_flops = base_model_flops + adapter_flops + classifier_flops
        
        # Estimate energy consumption (in arbitrary units)
        # Assuming energy is proportional to FLOPs
        energy_consumption = total_flops * 1e-9  # Convert to GFLOPs
        
        return {
            "num_trainable_params": num_trainable_params,
            "num_total_params": num_total_params,
            "trainable_params_percentage": num_trainable_params / num_total_params * 100,
            "estimated_gflops": total_flops * 1e-9,
            "estimated_energy": energy_consumption,
        }


class PaPIForSequenceClassification(nn.Module):
    """
    PaPI model for sequence classification tasks.
    
    This is a wrapper around the PaPI model that provides a simpler interface
    for sequence classification tasks.
    
    Args:
        num_tasks (int): Number of tasks
        num_labels_per_task (list): List of number of labels for each task
        pretrained_model_name (str): Name of the pretrained model to use
        adapter_bottleneck_dim (int): Dimension of the adapter bottleneck
        use_routing (bool): Whether to use the routing mechanism
        routing_temperature (float): Temperature parameter for the routing mechanism
        freeze_base_model (bool): Whether to freeze the base model parameters
    """
    def __init__(
        self,
        num_tasks,
        num_labels_per_task,
        pretrained_model_name="roberta-base",
        adapter_bottleneck_dim=64,
        use_routing=True,
        routing_temperature=1.0,
        freeze_base_model=True,
    ):
        super().__init__()
        self.model = PaPIModel(
            num_tasks=num_tasks,
            num_labels_per_task=num_labels_per_task,
            pretrained_model_name=pretrained_model_name,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            use_routing=use_routing,
            routing_temperature=routing_temperature,
            freeze_base_model=freeze_base_model,
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_id=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            token_type_ids (torch.Tensor, optional): Token type IDs
            task_id (int, optional): Task ID
            labels (torch.Tensor, optional): Labels for computing the loss
            
        Returns:
            dict: Dictionary containing model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_id=task_id,
        )
        
        logits = outputs["logits"]
        task_id = outputs["task_id"]
        routing_probs = outputs["routing_probs"]
        
        result = {
            "logits": logits,
            "task_id": task_id,
            "routing_probs": routing_probs,
        }
        
        # Compute loss if labels are provided
        if labels is not None:
            if isinstance(task_id, torch.Tensor):
                # Handle batch with different task IDs
                loss = 0
                for i, t_id in enumerate(task_id):
                    t_id = t_id.item()
                    num_labels = self.model.num_labels_per_task[t_id]
                    if num_labels == 1:
                        # Regression task
                        loss_fct = nn.MSELoss()
                        loss += loss_fct(logits[i].view(-1), labels[i].view(-1))
                    else:
                        # Classification task
                        loss_fct = nn.CrossEntropyLoss()
                        loss += loss_fct(logits[i].view(-1, num_labels), labels[i].view(-1))
                loss /= len(task_id)
            else:
                # Same task for the whole batch
                num_labels = self.model.num_labels_per_task[task_id]
                if num_labels == 1:
                    # Regression task
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    # Classification task
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
            result["loss"] = loss
        
        return result
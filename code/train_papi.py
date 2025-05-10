"""
Training script for the PaPI model.

This script demonstrates how to train the PaPI model on a sequence of tasks
(SST-2, Emotion, and MNLI) using the Elastic Weight Consolidation (EWC)
regularization technique to prevent catastrophic forgetting.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from models.papi_model import PaPIForSequenceClassification
from losses.ewc_regularization import EWCRegularization, EWCLoss
from utils.routing_utils import (
    train_routing_classifier,
    evaluate_routing_accuracy,
    track_energy_consumption,
    compute_forgetting_rate,
    compute_backward_transfer,
    compute_forward_transfer,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PaPI model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Adapter bottleneck dimension")
    parser.add_argument("--use_routing", action="store_true", help="Use routing mechanism")
    parser.add_argument("--routing_temp", type=float, default=1.0, help="Routing temperature")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base model parameters")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC regularization strength")
    parser.add_argument("--fisher_samples", type=int, default=500, help="Number of samples for Fisher computation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_sst2_dataset(tokenizer, max_length=128):
    """Load the SST-2 dataset."""
    dataset = load_dataset("glue", "sst2")
    
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset


def load_emotion_dataset(tokenizer, max_length=128):
    """Load the Emotion dataset."""
    dataset = load_dataset("emotion")
    
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["label"]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset


def load_mnli_dataset(tokenizer, max_length=128):
    """Load the MNLI dataset."""
    dataset = load_dataset("glue", "mnli")
    
    train_texts = [f"{premise} {tokenizer.sep_token} {hypothesis}" for premise, hypothesis in zip(dataset["train"]["premise"], dataset["train"]["hypothesis"])]
    train_labels = dataset["train"]["label"]
    val_texts = [f"{premise} {tokenizer.sep_token} {hypothesis}" for premise, hypothesis in zip(dataset["validation_matched"]["premise"], dataset["validation_matched"]["hypothesis"])]
    val_labels = dataset["validation_matched"]["label"]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset


def train_task(model, train_dataloader, optimizer, scheduler, device, task_id, ewc=None, ewc_lambda=1.0, num_epochs=3):
    """Train the model on a specific task."""
    model.train()
    
    # Define loss function
    if ewc is not None:
        # Use EWC loss
        task_loss_fn = nn.CrossEntropyLoss()
        loss_fn = EWCLoss(task_loss_fn, ewc, ewc_lambda)
    else:
        # Use standard cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                task_id=task_id,
                labels=batch["labels"],
            )
            
            # Compute loss
            if ewc is not None:
                # EWC loss is already computed in the forward pass
                loss = outputs["loss"] + ewc_lambda * ewc.penalty()
            else:
                loss = loss_fn(outputs["logits"], batch["labels"])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item()
        
        # Print epoch statistics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")
    
    return model


def evaluate_task(model, eval_dataloader, device, task_id):
    """Evaluate the model on a specific task."""
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                task_id=task_id,
            )
            
            # Compute accuracy
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_predictions += batch["labels"].size(0)
    
    accuracy = correct_predictions / total_predictions
    print(f"Task {task_id} Accuracy: {accuracy:.4f}")
    
    return accuracy


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    print("Loading datasets...")
    sst2_train, sst2_val = load_sst2_dataset(tokenizer, args.max_seq_length)
    emotion_train, emotion_val = load_emotion_dataset(tokenizer, args.max_seq_length)
    mnli_train, mnli_val = load_mnli_dataset(tokenizer, args.max_seq_length)
    
    # Create dataloaders
    sst2_train_dataloader = DataLoader(sst2_train, batch_size=args.batch_size, shuffle=True)
    sst2_val_dataloader = DataLoader(sst2_val, batch_size=args.batch_size)
    emotion_train_dataloader = DataLoader(emotion_train, batch_size=args.batch_size, shuffle=True)
    emotion_val_dataloader = DataLoader(emotion_val, batch_size=args.batch_size)
    mnli_train_dataloader = DataLoader(mnli_train, batch_size=args.batch_size, shuffle=True)
    mnli_val_dataloader = DataLoader(mnli_val, batch_size=args.batch_size)
    
    # Define tasks
    tasks = [
        {"name": "sst2", "train_dataloader": sst2_train_dataloader, "val_dataloader": sst2_val_dataloader, "num_labels": 2},
        {"name": "emotion", "train_dataloader": emotion_train_dataloader, "val_dataloader": emotion_val_dataloader, "num_labels": 6},
        {"name": "mnli", "train_dataloader": mnli_train_dataloader, "val_dataloader": mnli_val_dataloader, "num_labels": 3},
    ]
    
    # Create model
    print("Creating model...")
    model = PaPIForSequenceClassification(
        num_tasks=len(tasks),
        num_labels_per_task=[task["num_labels"] for task in tasks],
        pretrained_model_name=args.model_name,
        adapter_bottleneck_dim=args.adapter_dim,
        use_routing=args.use_routing,
        routing_temperature=args.routing_temp,
        freeze_base_model=args.freeze_base,
    )
    model.to(device)
    
    # Initialize EWC
    ewc = None
    
    # Track accuracies for computing backward and forward transfer
    accuracies = [[0.0 for _ in range(len(tasks))] for _ in range(len(tasks))]
    
    # Train on each task sequentially
    for task_idx, task in enumerate(tasks):
        print(f"\n=== Training on {task['name']} ===")
        
        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(task["train_dataloader"]) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
        
        # Train on current task
        model = train_task(
            model=model,
            train_dataloader=task["train_dataloader"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            task_id=task_idx,
            ewc=ewc,
            ewc_lambda=args.ewc_lambda,
            num_epochs=args.num_epochs,
        )
        
        # Evaluate on all tasks
        for eval_task_idx, eval_task in enumerate(tasks):
            print(f"\n--- Evaluating on {eval_task['name']} ---")
            accuracy = evaluate_task(
                model=model,
                eval_dataloader=eval_task["val_dataloader"],
                device=device,
                task_id=eval_task_idx,
            )
            accuracies[eval_task_idx][task_idx] = accuracy
        
        # Update EWC after training on the current task
        if task_idx == 0:
            # Initialize EWC after training on the first task
            print("\nInitializing EWC...")
            ewc = EWCRegularization(
                model=model,
                dataset=sst2_train,  # Use SST-2 for Fisher computation
                device=device,
                fisher_sample_size=args.fisher_samples,
                fisher_alpha=args.ewc_lambda,
            )
        elif ewc is not None:
            # Update EWC with data from the current task
            print("\nUpdating EWC...")
            ewc.update_fisher_and_means(task["train_dataloader"].dataset)
    
    # Compute backward and forward transfer
    backward_transfer = compute_backward_transfer(accuracies)
    forward_transfer = compute_forward_transfer(accuracies)
    
    print("\n=== Final Results ===")
    print(f"Backward Transfer: {backward_transfer:.4f}")
    print(f"Forward Transfer: {forward_transfer:.4f}")
    
    # Compute forgetting rates
    for task_idx, task in enumerate(tasks[:-1]):  # Exclude the last task
        initial_accuracy = accuracies[task_idx][task_idx]
        final_accuracy = accuracies[task_idx][-1]
        forgetting_rate = compute_forgetting_rate(model, task_idx, initial_accuracy, final_accuracy)
        print(f"Forgetting Rate for {task['name']}: {forgetting_rate:.4f}")
    
    # Track energy consumption
    energy_stats = track_energy_consumption(
        model=model.model,  # Access the underlying PaPIModel
        batch_size=args.batch_size,
        seq_length=args.max_seq_length,
        num_iterations=sum(len(task["train_dataloader"]) * args.num_epochs for task in tasks),
    )
    
    print("\n=== Energy Consumption ===")
    print(f"Trainable Parameters: {energy_stats['num_trainable_params']} ({energy_stats['trainable_params_percentage']:.2f}%)")
    print(f"Estimated GFLOPs: {energy_stats['estimated_gflops']:.4f}")
    print(f"Estimated Energy: {energy_stats['energy_kwh']:.6f} kWh")
    print(f"Estimated CO2 Emissions: {energy_stats['co2_emissions_g']:.2f} g")
    
    # Save model
    model_path = os.path.join(args.output_dir, "papi_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
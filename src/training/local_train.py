"""Local training utilities for LoRA fine-tuning with 4-bit quantization."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import yaml
import os
from pathlib import Path


class MedicalDataset(Dataset):
    """Dataset for pre-tokenized medical data."""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Use pre-tokenized data
        input_ids = item['input_ids'][:self.max_length]
        attention_mask = item['attention_mask'][:self.max_length]
        
        # Pad to max_length for batching
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM
        }


def train_local_model(
    model,
    train_data,
    tokenizer,
    output_dir="./output-models/local-lora",
    num_steps=100,
    batch_size=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    max_length=2048,
    max_samples=None,
    gpu_id=None
):
    """
    Train model locally with LoRA adapters using custom training loop.
    
    Args:
        model: Model with LoRA adapters (already quantized and prepared)
        train_data: List of pre-tokenized samples with 'input_ids' and 'attention_mask'
        tokenizer: Tokenizer for the model
        output_dir: Directory to save checkpoints
        num_steps: Number of training steps
        batch_size: Batch size (default 1 for memory efficiency)
        learning_rate: Learning rate
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to use from train_data (None = use all)
        gpu_id: GPU ID to use (e.g., 0, 1, 2, 3). None = use default
    
    Returns:
        dict: Training metrics (losses, final model state)
    """
    # Set GPU if specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🎮 Using GPU {gpu_id}")
    
    # Limit samples if max_samples is specified
    if max_samples is not None and max_samples < len(train_data):
        train_data = train_data[:max_samples]
        print(f"📊 Using {max_samples} samples from dataset")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    train_dataset = MedicalDataset(train_data, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training setup
    model.train()
    losses = []
    vram_usage = []
    
    print(f"🚀 Starting local training...")
    print(f"   Dataset: {len(train_dataset)} samples")
    print(f"   Steps: {num_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Output dir: {output_dir}\n")
    
    step = 0
    iterator = iter(train_loader)
    
    with tqdm(total=num_steps, desc="Training") as pbar:
        while step < num_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                # Reset iterator when dataset is exhausted
                iterator = iter(train_loader)
                batch = next(iterator)
            
            # Move to GPU
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            losses.append(loss.item())
            if torch.cuda.is_available():
                vram_usage.append(torch.cuda.memory_allocated(0)/1024**3)
            
            # Periodic logging
            if (step + 1) % logging_steps == 0:
                avg_loss = np.mean(losses[-logging_steps:])
                current_vram = vram_usage[-1] if vram_usage else 0
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'vram': f'{current_vram:.2f}GB'
                })
            
            # Save checkpoint
            if (step + 1) % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step+1}")
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\n💾 Checkpoint saved: {checkpoint_dir}")
            
            step += 1
            pbar.update(1)
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Training summary
    print(f"\n{'='*60}")
    print("✅ Training complete!")
    print(f"{'='*60}")
    print(f"📉 Initial loss: {losses[0]:.4f}")
    print(f"📉 Final loss: {losses[-1]:.4f}")
    print(f"📉 Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    if vram_usage:
        print(f"💾 Peak VRAM: {max(vram_usage):.2f} GB")
    print(f"💾 Model saved: {final_dir}")
    print(f"{'='*60}\n")
    
    return {
        'losses': losses,
        'vram_usage': vram_usage,
        'final_loss': losses[-1],
        'model_path': final_dir
    }

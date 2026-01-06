"""Federated learning client implementation using Flower."""

import flwr as fl
from typing import Dict, Tuple, List
import torch
import numpy as np
from collections import OrderedDict
import json
import os


class MedicalFLClient(fl.client.NumPyClient):
    """
    Flower federated learning client for medical AI with LoRA.
    
    Only transmits LoRA adapter weights (not full model).
    Trains locally and returns deltas + metrics.
    """
    
    def __init__(
        self, 
        model,  # Model with LoRA adapters
        train_data: List[Dict],  # Pre-tokenized training data
        tokenizer,
        hospital_name: str,
        local_train_fn,  # Training function from local_train.py
        num_steps: int = 100,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        max_length: int = 2048
    ):
        """
        Initialize federated client.
        
        Args:
            model: Base model with LoRA adapters already attached
            train_data: List of pre-tokenized samples
            tokenizer: Model tokenizer
            hospital_name: Hospital/client identifier
            local_train_fn: Training function (from local_train.py)
            num_steps: Number of local training steps
            batch_size: Batch size for training
            learning_rate: Learning rate
            max_length: Maximum sequence length
        """
        self.model = model
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.hospital_name = hospital_name
        self.local_train_fn = local_train_fn
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.num_samples = len(train_data)
        
        print(f"✅ Client '{hospital_name}' initialized")
        print(f"   - Samples: {self.num_samples:,}")
        print(f"   - Training steps: {num_steps}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Extract LoRA adapter parameters only.
        
        Returns:
            List of numpy arrays containing only LoRA weights
        """
        lora_state_dict = self._get_lora_state_dict()
        return [val.cpu().numpy() for val in lora_state_dict.values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set LoRA adapter parameters from global model.
        
        Args:
            parameters: List of numpy arrays with LoRA weights
        """
        lora_keys = [k for k in self.model.state_dict().keys() if 'lora' in k.lower()]
        
        if len(parameters) != len(lora_keys):
            print(f"⚠️  Warning: Parameter count mismatch!")
            print(f"   Expected {len(lora_keys)}, got {len(parameters)}")
            return
        
        params_dict = zip(lora_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ Loaded global LoRA parameters ({len(parameters)} tensors)")
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally and return LoRA deltas.
        
        Args:
            parameters: Global LoRA parameters
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        print(f"\n{'='*60}")
        print(f"🏥 CLIENT: {self.hospital_name}")
        print(f"{'='*60}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Train locally
        print(f"🔄 Starting local training ({self.num_steps} steps)...")
        results = self.local_train_fn(
            model=self.model,
            train_data=self.train_data,
            tokenizer=self.tokenizer,
            output_dir=f"output-models/federated/{self.hospital_name}",
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            logging_steps=max(1, self.num_steps // 10),
            save_steps=self.num_steps  # Only save at end
        )
        
        # Extract updated LoRA parameters
        updated_parameters = self.get_parameters(config={})
        
        # Prepare metrics
        metrics = {
            "hospital": self.hospital_name,
            "num_samples": self.num_samples,
            "initial_loss": float(results['losses'][0]),
            "final_loss": float(results['final_loss']),
            "loss_reduction": float(
                (results['losses'][0] - results['final_loss']) / results['losses'][0] * 100
            ),
            "steps": self.num_steps
        }
        
        if results['vram_usage']:
            metrics["peak_vram_gb"] = float(max(results['vram_usage']))
        
        print(f"\n📊 Training metrics:")
        print(f"   - Loss: {metrics['initial_loss']:.4f} → {metrics['final_loss']:.4f}")
        print(f"   - Improvement: {metrics['loss_reduction']:.2f}%")
        print(f"   - Samples: {self.num_samples:,}")
        
        # Calculate LoRA weight statistics
        total_params = sum([p.size for p in updated_parameters])
        total_bytes = sum([p.nbytes for p in updated_parameters])
        metrics["lora_params_transmitted"] = int(total_params)
        metrics["lora_size_mb"] = float(total_bytes / (1024**2))
        
        print(f"\n📤 Transmission:")
        print(f"   - LoRA params: {metrics['lora_params_transmitted']:,}")
        print(f"   - Size: {metrics['lora_size_mb']:.2f} MB")
        print(f"{'='*60}\n")
        
        return updated_parameters, self.num_samples, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data.
        
        Args:
            parameters: Global LoRA parameters
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Use small subset for quick evaluation
        eval_samples = min(100, len(self.train_data))
        eval_data = self.train_data[:eval_samples]
        
        self.model.eval()
        total_loss = 0.0
        
        from src.training.local_train import MedicalDataset
        from torch.utils.data import DataLoader
        
        eval_dataset = MedicalDataset(eval_data, self.tokenizer, self.max_length)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(eval_loader)
        
        metrics = {
            "hospital": self.hospital_name,
            "eval_loss": float(avg_loss),
            "eval_samples": eval_samples
        }
        
        return avg_loss, eval_samples, metrics
    
    def _get_lora_state_dict(self) -> OrderedDict:
        """
        Extract only LoRA adapter weights from model.
        
        Returns:
            OrderedDict with LoRA parameters only
        """
        lora_state_dict = OrderedDict()
        for name, param in self.model.state_dict().items():
            if 'lora' in name.lower():
                lora_state_dict[name] = param
        
        return lora_state_dict

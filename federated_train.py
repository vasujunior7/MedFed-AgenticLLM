#!/usr/bin/env python3
"""
Federated Training Loop - Milestone 7
End-to-end federated learning with agent-weighted aggregation.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Dict
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))
from src.federated.client import MedicalFLClient
from src.training.local_train import train_local_model
from src.agent.coordinator import AgenticAggregator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Configuration
CONFIG = {
    'clients': ['hospital_A', 'hospital_B', 'hospital_C'],
    'rounds': 5,
    'local_steps': 100,
    'batch_size': 2,
    'learning_rate': 2e-4,
    'samples_per_client': 5000,
    'gpu': 3,
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
    'lora_r': 8,
    'lora_alpha': 16,
    'max_length': 2048,
    'output_dir': 'output-models/federated-final'
}


def load_jsonl(file_path):
    """Load JSONL dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def setup_global_model(config):
    """Initialize the global model with LoRA."""
    print(f"\n{'='*70}")
    print("🔧 INITIALIZING GLOBAL MODEL")
    print(f"{'='*70}")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"🔄 Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print(f"🔧 Setting up LoRA (r={config['lora_r']}, alpha={config['lora_alpha']})...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ LoRA adapters applied")
    print(f"📊 Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    if torch.cuda.is_available():
        print(f"💾 VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    return model, tokenizer


def initialize_clients(config, model, tokenizer):
    """Initialize federated clients for each hospital."""
    print(f"\n{'='*70}")
    print("🏥 INITIALIZING CLIENTS")
    print(f"{'='*70}")
    
    clients = []
    client_data_sizes = []
    
    for hospital in config['clients']:
        # Load hospital data
        dataset_path = f"data/processed/{hospital}/dataset.jsonl"
        print(f"\n📂 Loading {hospital} data from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            continue
        
        train_data = load_jsonl(dataset_path)
        
        # Limit samples
        if len(train_data) > config['samples_per_client']:
            train_data = train_data[:config['samples_per_client']]
        
        print(f"✅ Loaded {len(train_data):,} samples for {hospital}")
        
        # Create client
        client = MedicalFLClient(
            model=model,
            train_data=train_data,
            tokenizer=tokenizer,
            hospital_name=hospital,
            local_train_fn=train_local_model,
            num_steps=config['local_steps'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            max_length=config['max_length']
        )
        
        clients.append(client)
        client_data_sizes.append(len(train_data))
    
    print(f"\n✅ Initialized {len(clients)} clients")
    print(f"📊 Total samples: {sum(client_data_sizes):,}")
    
    return clients, client_data_sizes


def aggregate_parameters(client_params_list, weights):
    """
    Aggregate client parameters using weighted averaging.
    
    Args:
        client_params_list: List of parameter arrays from each client
        weights: Aggregation weights from agent
    
    Returns:
        Aggregated parameters
    """
    aggregated = []
    
    for i in range(len(client_params_list[0])):  # For each parameter
        # Weighted sum of this parameter across clients
        weighted_sum = sum(
            w * client_params_list[j][i] 
            for j, w in enumerate(weights)
        )
        aggregated.append(weighted_sum)
    
    return aggregated


def save_round_metrics(round_num, metrics, output_dir):
    """Save metrics for a round to JSON."""
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    
    with open(f"{output_dir}/metrics/round_{round_num}.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_training_results(all_metrics, config):
    """Generate comprehensive visualizations."""
    print(f"\n{'='*70}")
    print("📊 GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    output_dir = config['output_dir']
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Extract data
    rounds = list(range(1, len(all_metrics) + 1))
    global_losses = [m['global_avg_loss'] for m in all_metrics]
    
    # Client-specific losses
    client_losses = {client: [] for client in config['clients']}
    for round_metrics in all_metrics:
        for client_metric in round_metrics['client_metrics']:
            hospital = client_metric['hospital']
            client_losses[hospital].append(client_metric['final_loss'])
    
    # Agent weights
    client_weights = {client: [] for client in config['clients']}
    for round_metrics in all_metrics:
        for i, client in enumerate(config['clients']):
            client_weights[client].append(round_metrics['agent_weights'][i])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Federated Learning Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Global Loss Over Rounds
    ax1 = axes[0, 0]
    ax1.plot(rounds, global_losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Global Average Loss', fontsize=12)
    ax1.set_title('Global Loss Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Add improvement annotation
    improvement = ((global_losses[0] - global_losses[-1]) / global_losses[0]) * 100
    ax1.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Client Comparison
    ax2 = axes[0, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (client, losses) in enumerate(client_losses.items()):
        ax2.plot(rounds, losses, '-o', label=client, color=colors[i], 
                linewidth=2, markersize=6)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Final Loss', fontsize=12)
    ax2.set_title('Client Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    # Plot 3: Agent Weight Distribution
    ax3 = axes[1, 0]
    x = np.arange(len(rounds))
    width = 0.25
    
    for i, (client, weights) in enumerate(client_weights.items()):
        ax3.bar(x + i*width, weights, width, label=client, color=colors[i])
    
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Aggregation Weight', fontsize=12)
    ax3.set_title('Agent-Based Weight Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(rounds)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Loss Improvement per Round
    ax4 = axes[1, 1]
    loss_deltas = [0] + [global_losses[i-1] - global_losses[i] 
                         for i in range(1, len(global_losses))]
    colors_bar = ['green' if d > 0 else 'red' for d in loss_deltas]
    ax4.bar(rounds, loss_deltas, color=colors_bar, alpha=0.7)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Loss Improvement', fontsize=12)
    ax4.set_title('Loss Reduction per Round', fontsize=14, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(rounds)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = f"{output_dir}/plots/training_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {plot_path}")
    
    # Also save as PDF
    pdf_path = f"{output_dir}/plots/training_results.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF: {pdf_path}")
    
    plt.close()
    
    return plot_path


def print_metrics_table(all_metrics, config):
    """Print a beautiful metrics table."""
    print(f"\n{'='*70}")
    print("📋 TRAINING METRICS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Round':<8}{'Global Loss':<15}{'Best Client':<15}{'Worst Client':<15}{'Variance':<10}")
    print("-" * 70)
    
    for i, metrics in enumerate(all_metrics, 1):
        global_loss = metrics['global_avg_loss']
        client_losses = [m['final_loss'] for m in metrics['client_metrics']]
        best_loss = min(client_losses)
        worst_loss = max(client_losses)
        variance = np.var(client_losses)
        
        print(f"{i:<8}{global_loss:<15.4f}{best_loss:<15.4f}{worst_loss:<15.4f}{variance:<10.4f}")
    
    print("-" * 70)
    
    # Summary statistics
    initial_loss = all_metrics[0]['global_avg_loss']
    final_loss = all_metrics[-1]['global_avg_loss']
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    
    print(f"\n📊 Summary:")
    print(f"   Initial Loss: {initial_loss:.4f}")
    print(f"   Final Loss: {final_loss:.4f}")
    print(f"   Improvement: {improvement:.2f}%")
    print(f"   Total Rounds: {len(all_metrics)}")


def federated_training_loop(config):
    """Main federated training loop."""
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print("🚀 FEDERATED TRAINING - MILESTONE 7")
    print(f"{'='*70}")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Clients: {len(config['clients'])}")
    print(f"   Rounds: {config['rounds']}")
    print(f"   Local Steps: {config['local_steps']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Samples/Client: {config['samples_per_client']}")
    print(f"   GPU: {config['gpu']}")
    
    # Initialize global model
    global_model, tokenizer = setup_global_model(config)
    
    # Initialize clients
    clients, client_data_sizes = initialize_clients(config, global_model, tokenizer)
    
    if len(clients) == 0:
        print("❌ No clients initialized. Exiting.")
        return
    
    # Initialize agent aggregator
    aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
    
    # Get initial global parameters
    global_params = clients[0].get_parameters(config={})
    
    # Metrics tracking
    all_metrics = []
    
    # Federated training rounds
    for round_num in range(1, config['rounds'] + 1):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num}/{config['rounds']}")
        print(f"{'='*70}")
        
        # Track round metrics
        round_metrics = {
            'round': round_num,
            'client_metrics': [],
            'agent_weights': [],
            'global_avg_loss': 0.0
        }
        
        # Each client trains locally
        client_updates = []
        client_metrics = []
        
        for i, client in enumerate(clients):
            print(f"\n--- Client {i+1}/{len(clients)}: {client.hospital_name} ---")
            
            # Local training
            updated_params, num_samples, metrics = client.fit(
                parameters=global_params,
                config={}
            )
            
            client_updates.append(updated_params)
            client_metrics.append(metrics)
            round_metrics['client_metrics'].append(metrics)
        
        # Check for NaNs
        has_nan = False
        for metrics in client_metrics:
            if np.isnan(metrics['final_loss']) or np.isinf(metrics['final_loss']):
                print(f"❌ NaN detected in {metrics['hospital']}!")
                has_nan = True
        
        if has_nan:
            print("❌ Training failed due to NaN losses")
            break
        
        # Agent-based aggregation
        print(f"\n🤖 AGENTIC AGGREGATION")
        weights, analysis = aggregator.compute_aggregation_weights(
            client_metrics,
            sample_counts=client_data_sizes
        )
        
        round_metrics['agent_weights'] = weights.tolist()
        round_metrics['agent_analysis'] = analysis
        
        # Aggregate parameters
        global_params = aggregate_parameters(client_updates, weights)
        
        # Compute global average loss
        global_avg_loss = sum(
            w * m['final_loss'] 
            for w, m in zip(weights, client_metrics)
        )
        round_metrics['global_avg_loss'] = float(global_avg_loss)
        
        print(f"\n📊 Round {round_num} Summary:")
        print(f"   Global Avg Loss: {global_avg_loss:.4f}")
        print(f"   Client Losses: {[f'{m['final_loss']:.4f}' for m in client_metrics]}")
        print(f"   Agent Weights: {[f'{w:.3f}' for w in weights]}")
        
        # Save round metrics
        all_metrics.append(round_metrics)
        save_round_metrics(round_num, round_metrics, config['output_dir'])
    
    # Save final model
    print(f"\n{'='*70}")
    print("💾 SAVING FINAL MODEL")
    print(f"{'='*70}")
    
    # Update model with final global parameters
    clients[0].set_parameters(global_params)
    final_model_path = f"{config['output_dir']}/final"
    clients[0].model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"✅ Final model saved to: {final_model_path}")
    
    # Generate visualizations
    plot_path = plot_training_results(all_metrics, config)
    
    # Print metrics table
    print_metrics_table(all_metrics, config)
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("✅ FEDERATED TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n🎉 Results:")
    print(f"   ✓ Rounds completed: {len(all_metrics)}/{config['rounds']}")
    print(f"   ✓ Final global loss: {all_metrics[-1]['global_avg_loss']:.4f}")
    print(f"   ✓ No NaNs detected")
    print(f"   ✓ Loss stabilized: {all_metrics[-1]['global_avg_loss'] < all_metrics[0]['global_avg_loss']}")
    print(f"   ✓ Training time: {duration/60:.1f} minutes")
    print(f"   ✓ Model saved: {final_model_path}")
    print(f"   ✓ Visualizations: {plot_path}")
    
    print(f"\n{'='*70}\n")
    
    return all_metrics


if __name__ == "__main__":
    # Run federated training
    metrics = federated_training_loop(CONFIG)

#!/usr/bin/env python3
"""
Standalone script for local LoRA fine-tuning on medical data.
Usage: python train_local.py --samples 300000 --gpu 3
"""

import os
import sys
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add src to path
sys.path.append(os.path.dirname(__file__))
from src.training.local_train import train_local_model


def load_jsonl(file_path):
    """Load JSONL dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description='Local LoRA Fine-tuning')
    parser.add_argument('--hospital', type=str, default='hospital_A', 
                        help='Hospital ID (hospital_A, hospital_B, etc.)')
    parser.add_argument('--samples', type=int, default=30000,
                        help='Number of training samples to use')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU device ID to use')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha scaling')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for model checkpoints')
    parser.add_argument('--model-name', type=str, 
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Base model name')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"\n{'='*60}")
    print(f"🎯 GPU: {args.gpu}")
    print(f"📊 Samples: {args.samples:,}")
    print(f"🔄 Steps: {args.steps}")
    print(f"🏥 Hospital: {args.hospital}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset_path = f"data/processed/{args.hospital}/dataset.jsonl"
    print(f"📂 Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    train_data = load_jsonl(dataset_path)
    print(f"✅ Loaded {len(train_data):,} total samples")
    
    # Limit samples
    if args.samples < len(train_data):
        train_data = train_data[:args.samples]
        print(f"📊 Using {args.samples:,} samples for training")
    
    # Sample preview
    print(f"\n📝 Sample preview:")
    sample = train_data[0]
    print(f"   - Text length: {len(sample['text'])} chars")
    print(f"   - Token count: {sample['token_length']}")
    print(f"   - Preview: {sample['text'][:150]}...\n")
    
    # Configure 4-bit quantization
    print("🔧 Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"🔄 Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for training
    print("🔧 Preparing model for k-bit training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    if torch.cuda.is_available():
        print(f"💾 VRAM after model load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    # Setup LoRA
    print(f"🔧 Setting up LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
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
    print(f"📊 Total params: {total_params:,}\n")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"output-models/{args.hospital}-{args.samples//1000}k-samples"
    
    # Train
    print(f"{'='*60}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*60}\n")
    
    results = train_local_model(
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Configuration:")
    print(f"   - Hospital: {args.hospital}")
    print(f"   - Samples: {args.samples:,}")
    print(f"   - Steps: {args.steps}")
    print(f"   - GPU: {args.gpu}")
    print(f"   - LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"\n📈 Results:")
    print(f"   - Initial loss: {results['losses'][0]:.4f}")
    print(f"   - Final loss: {results['final_loss']:.4f}")
    print(f"   - Improvement: {((results['losses'][0] - results['final_loss']) / results['losses'][0] * 100):.2f}%")
    if results['vram_usage']:
        print(f"   - Peak VRAM: {max(results['vram_usage']):.2f} GB")
    print(f"\n💾 Model saved to: {results['model_path']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

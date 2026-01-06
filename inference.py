#!/usr/bin/env python3
"""
Inference CLI - Milestone 8
Loads final LoRA adapter and generates safe medical responses.

Usage:
    python inference.py "I have chest pain and fatigue"
    python inference.py --hospital hospital_B "What causes diabetes?"
    python inference.py --federated "Symptoms of flu?"
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add src to path
sys.path.append(os.path.dirname(__file__))
from src.safety.guardrails import MedicalGuardrails


def format_medical_prompt(query: str) -> str:
    """Format user query into medical Q&A prompt."""
    prompt = f"""[INST] You are a medical AI assistant. Answer the following medical question accurately and helpfully.

Question: {query}

Provide a clear, informative response. [/INST]"""
    return prompt


def load_model_with_lora(adapter_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2", gpu: int = 3):
    """
    Load base model with LoRA adapter.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        base_model: Base model name
        gpu: GPU device ID
        
    Returns:
        model, tokenizer
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"🔧 Loading model from {base_model}")
    print(f"📂 LoRA adapter: {adapter_path}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapter
    print(f"🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    if torch.cuda.is_available():
        print(f"💾 VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    """
    Generate response from model.
    
    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer
        prompt: Formatted prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (exclude prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response after [/INST]
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response
    
    return response


def main():
    parser = argparse.ArgumentParser(description='Medical AI Inference CLI')
    parser.add_argument('query', type=str, nargs='?', default=None,
                        help='Medical query (e.g., "I have chest pain and fatigue")')
    parser.add_argument('--hospital', type=str, default=None,
                        help='Use specific hospital model (hospital_A, hospital_B, hospital_C)')
    parser.add_argument('--federated', action='store_true',
                        help='Use federated aggregated model (default)')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU device ID (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Maximum tokens to generate (default: 256)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--no-disclaimer', action='store_true',
                        help='Skip adding medical disclaimer')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Base model name')
    
    args = parser.parse_args()
    
    # Get query
    if args.query is None:
        print("❌ Error: Please provide a medical query")
        print("Usage: python inference.py \"I have chest pain and fatigue\"")
        sys.exit(1)
    
    query = args.query
    
    # Determine adapter path
    if args.hospital:
        # Use specific hospital model
        adapter_path = f"output-models/federated/{args.hospital}/final"
        model_name = f"Hospital {args.hospital[-1]} Model"
    else:
        # Try to use federated model, fallback to hospital_B (best performer)
        federated_path = "output-models/federated/global/final"
        if os.path.exists(federated_path):
            adapter_path = federated_path
            model_name = "Federated Global Model"
        else:
            # Use hospital_B (best performer from Round 3)
            adapter_path = "output-models/federated/hospital_B/final"
            model_name = "Hospital B Model (Best Performer)"
            print(f"ℹ️  Federated global model not found, using {model_name}")
    
    # Verify adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ Error: LoRA adapter not found at {adapter_path}")
        print(f"\nAvailable models:")
        for hospital in ['hospital_A', 'hospital_B', 'hospital_C']:
            path = f"output-models/federated/{hospital}/final"
            if os.path.exists(path):
                print(f"   ✓ {hospital}: {path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🏥 MEDICAL AI INFERENCE - MILESTONE 8")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Adapter: {adapter_path}")
    print(f"   GPU: {args.gpu}")
    print(f"   Max Tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"\n❓ Query: {query}")
    print("\n" + "-"*70)
    
    # Load model
    print("\n🔄 Loading model...")
    model, tokenizer = load_model_with_lora(
        adapter_path=adapter_path,
        base_model=args.model,
        gpu=args.gpu
    )
    print("✅ Model loaded successfully")
    
    # Format prompt
    prompt = format_medical_prompt(query)
    
    # Generate response
    print("\n🤖 Generating response...")
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Apply safety guardrails
    guardrails = MedicalGuardrails()
    safety_check = guardrails.check_response(response)
    
    # Add disclaimer if not disabled
    if not args.no_disclaimer:
        response = guardrails.add_disclaimer(response)
    
    # Display response
    print("\n" + "="*70)
    print("💬 RESPONSE:")
    print("="*70)
    print(response)
    print("\n" + "="*70)
    
    # Safety status
    if safety_check['is_safe']:
        print("✅ Safety Check: PASSED")
    else:
        print("⚠️  Safety Check: WARNINGS DETECTED")
        if safety_check['blocked_patterns']:
            print(f"   Blocked patterns: {safety_check['blocked_patterns']}")
    
    if safety_check['warnings']:
        print(f"⚠️  Warnings: {', '.join(safety_check['warnings'])}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

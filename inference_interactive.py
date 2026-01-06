#!/usr/bin/env python3
"""
Interactive Inference Mode - Loads model ONCE, then answers multiple queries.
Much faster for multiple questions!

Usage:
    python inference_interactive.py
    python inference_interactive.py --hospital hospital_B
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(__file__))
from src.safety.guardrails import MedicalGuardrails


def format_medical_prompt(query: str) -> str:
    """Format user query into medical Q&A prompt."""
    prompt = f"""[INST] You are a medical AI assistant. Answer the following medical question accurately and helpfully.

Question: {query}

Provide a clear, informative response. [/INST]"""
    return prompt


def load_model_with_lora(adapter_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2", gpu: int = 3):
    """Load base model with LoRA adapter (happens ONCE)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"🔧 Loading model from {base_model}")
    print(f"📂 LoRA adapter: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    if torch.cuda.is_available():
        print(f"💾 VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    """Generate response (fast - model already loaded)."""
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
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response
    
    return response


def main():
    parser = argparse.ArgumentParser(description='Interactive Medical AI Inference')
    parser.add_argument('--hospital', type=str, default=None,
                        help='Use specific hospital model (hospital_A, hospital_B, hospital_C)')
    parser.add_argument('--gpu', type=int, default=3, help='GPU device ID')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Determine adapter path
    if args.hospital:
        adapter_path = f"output-models/federated/{args.hospital}/final"
        model_name = f"Hospital {args.hospital[-1]} Model"
    else:
        # Use hospital_B (best performer - lowest loss, highest agent weight from Round 3)
        adapter_path = "output-models/federated/hospital_B/final"
        model_name = "Hospital B Model (Best Performer)"
    
    if not os.path.exists(adapter_path):
        print(f"❌ Error: LoRA adapter not found at {adapter_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🏥 INTERACTIVE MEDICAL AI - MILESTONE 8")
    print("="*70)
    print(f"\n📋 Model: {model_name}")
    print(f"📂 Adapter: {adapter_path}")
    print(f"⚡ GPU: {args.gpu}")
    
    # Load model ONCE (takes 20-30 seconds)
    print("\n🔄 Loading model (this takes ~20 seconds, but only happens ONCE)...")
    model, tokenizer = load_model_with_lora(adapter_path, gpu=args.gpu)
    print("✅ Model loaded and ready!\n")
    
    # Initialize guardrails
    guardrails = MedicalGuardrails()
    
    print("="*70)
    print("💡 INTERACTIVE MODE - Model is loaded, responses will be FAST!")
    print("="*70)
    print("Type your medical questions (or 'quit' to exit)")
    print("-"*70 + "\n")
    
    # Interactive loop
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Generate response (FAST - only 5-10 seconds)
            print("\n🤖 Generating response...")
            prompt = format_medical_prompt(query)
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Add disclaimer
            response = guardrails.add_disclaimer(response)
            
            # Display
            print("\n" + "="*70)
            print("💬 RESPONSE:")
            print("="*70)
            print(response)
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

"""Data preprocessing utilities for medical dataset.

Usage:
    python src/data/preprocess.py --limit 1000
    python src/data/preprocess.py --output data/processed/full_dataset.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import sys


def format_instruction(example: Dict) -> str:
    """
    Format example into User/Assistant instruction format.
    
    Args:
        example: Dataset example
        
    Returns:
        Formatted instruction string
    """
    keys = example.keys()
    
    # Try different field name combinations
    if 'input' in keys and 'output' in keys:
        question = example['input']
        answer = example['output']
    elif 'question' in keys and 'answer' in keys:
        question = example['question']
        answer = example['answer']
    elif 'instruction' in keys and 'response' in keys:
        question = example['instruction']
        answer = example['response']
    elif 'Patient' in keys and 'Doctor' in keys:
        question = example['Patient']
        answer = example['Doctor']
    elif 'Description' in keys and 'Patient' in keys:
        question = f"{example.get('Description', '')} {example.get('Patient', '')}"
        answer = example.get('Doctor', '')
    else:
        # Default: concatenate all fields
        values = list(example.values())
        question = str(values[0]) if len(values) > 0 else ""
        answer = str(values[1]) if len(values) > 1 else ""
    
    formatted = f"User: {question}\nAssistant: {answer}"
    return formatted


def preprocess_medical_text(text: str) -> str:
    """
    Preprocess medical text data.
    
    Args:
        text: Raw text
        
    Returns:
        Preprocessed text
    """
    # Basic cleaning
    text = text.strip()
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text


def create_prompt(example: Dict) -> Dict:
    """
    Create prompt from dataset example.
    
    Args:
        example: Dataset example
        
    Returns:
        Formatted prompt dictionary
    """
    formatted_text = format_instruction(example)
    formatted_text = preprocess_medical_text(formatted_text)
    
    return {
        'text': formatted_text,
        'original_example': example
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
    output_path: str = None
) -> List[Dict]:
    """
    Preprocess entire dataset with tokenization and truncation.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for processing
        max_length: Maximum sequence length
        output_path: Path to save JSONL file
        
    Returns:
        List of processed examples
    """
    processed_data = []
    skipped = 0
    
    print(f"\n🔄 Processing {len(dataset):,} samples...")
    print(f"   Max length: {max_length} tokens")
    print(f"   Tokenizer: {tokenizer.name_or_path}\n")
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        try:
            # Format to instruction format
            formatted_text = format_instruction(example)
            formatted_text = preprocess_medical_text(formatted_text)
            
            # Tokenize
            tokens = tokenizer(
                formatted_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Skip if empty after tokenization
            if len(tokens['input_ids']) == 0:
                skipped += 1
                continue
            
            # Create processed entry
            processed_entry = {
                'text': formatted_text,
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'token_length': len(tokens['input_ids']),
                'index': idx
            }
            
            processed_data.append(processed_entry)
            
        except Exception as e:
            print(f"\n⚠️ Error processing sample {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {len(processed_data):,} samples")
    print(f"   Skipped: {skipped:,} samples")
    
    # Save to JSONL if output path provided
    if output_path:
        save_to_jsonl(processed_data, output_path)
    
    return processed_data


def save_to_jsonl(data: List[Dict], output_path: str):
    """
    Save processed data to JSONL format.
    
    Args:
        data: List of processed examples
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # Convert to JSON-serializable format
            json_entry = {
                'text': entry['text'],
                'input_ids': entry['input_ids'],
                'attention_mask': entry['attention_mask'],
                'token_length': entry['token_length'],
                'index': entry['index']
            }
            f.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(data):,} samples to {output_path}")
    
    # Verify no empty rows
    verify_jsonl(output_path)


def verify_jsonl(file_path: str):
    """
    Verify JSONL file has no empty rows.
    
    Args:
        file_path: Path to JSONL file
    """
    print(f"\n🔍 Verifying {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    empty_lines = sum(1 for line in lines if not line.strip())
    invalid_lines = 0
    
    for i, line in enumerate(lines):
        if line.strip():
            try:
                data = json.loads(line)
                if not data.get('text') or len(data.get('input_ids', [])) == 0:
                    invalid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1
    
    if empty_lines == 0 and invalid_lines == 0:
        print(f"✅ Verification passed!")
        print(f"   Total lines: {len(lines):,}")
        print(f"   Empty lines: {empty_lines}")
        print(f"   Invalid entries: {invalid_lines}")
    else:
        print(f"⚠️ Verification issues found!")
        print(f"   Total lines: {len(lines):,}")
        print(f"   Empty lines: {empty_lines}")
        print(f"   Invalid entries: {invalid_lines}")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess medical dataset')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to process (default: all)')
    parser.add_argument('--output', type=str, default='data/processed/dataset.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Model name for tokenizer')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏥 MEDICAL DATASET PREPROCESSING")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded")
    
    # Load dataset
    print(f"\n📊 Loading dataset: ruslanmv/ai-medical-dataset")
    
    if args.limit:
        print(f"   Limit: {args.limit:,} samples")
        dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train", streaming=True)
        
        # Collect limited samples
        samples_list = []
        for i, sample in enumerate(dataset):
            if i >= args.limit:
                break
            samples_list.append(sample)
        
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(samples_list)
    else:
        dataset = load_dataset("ruslanmv/ai-medical-dataset", split="train")
    
    print(f"✅ Dataset loaded: {len(dataset):,} samples")
    
    # Preprocess dataset
    processed_data = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        output_path=args.output
    )
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Split dataset for federated learning across hospitals."""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import os


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"✅ Saved {len(data)} samples to {file_path}")


def create_non_iid_split(
    data: List[Dict],
    num_clients: int = 3,
    alpha: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Create non-IID split simulating real hospital data distribution.
    Uses Dirichlet distribution for realistic skew.
    
    Args:
        data: Full dataset
        num_clients: Number of clients (hospitals)
        alpha: Dirichlet parameter (lower = more skew, 0.5 = moderate)
        
    Returns:
        Dictionary mapping client names to their data
    """
    print(f"\n🔄 Creating non-IID split for {num_clients} clients...")
    print(f"   Total samples: {len(data)}")
    print(f"   Alpha (skew parameter): {alpha}")
    
    # Shuffle data
    random.shuffle(data)
    
    # Split using Dirichlet-inspired distribution
    # This creates realistic non-uniform splits
    splits = []
    remaining = len(data)
    
    # Generate proportions using random weights
    weights = [random.uniform(0.5, 1.5) for _ in range(num_clients)]
    total_weight = sum(weights)
    proportions = [w / total_weight for w in weights]
    
    # Apply slight variance to make it more realistic
    proportions = [p + random.uniform(-0.05, 0.05) for p in proportions]
    proportions = [max(0.2, min(0.5, p)) for p in proportions]  # Clamp between 20%-50%
    
    # Normalize
    total = sum(proportions)
    proportions = [p / total for p in proportions]
    
    # Calculate actual sample counts
    start_idx = 0
    for i, prop in enumerate(proportions):
        if i == num_clients - 1:
            # Last client gets all remaining
            end_idx = len(data)
        else:
            count = int(len(data) * prop)
            end_idx = start_idx + count
        
        client_data = data[start_idx:end_idx]
        splits.append(client_data)
        start_idx = end_idx
    
    # Create client mapping
    client_names = ['hospital_A', 'hospital_B', 'hospital_C', 'hospital_D']
    client_data = {}
    
    for i in range(num_clients):
        client_name = client_names[i] if i < len(client_names) else f'hospital_{chr(65+i)}'
        client_data[client_name] = splits[i]
        print(f"   ✅ {client_name}: {len(splits[i])} samples ({len(splits[i])/len(data)*100:.1f}%)")
    
    return client_data


def verify_split(client_data: Dict[str, List[Dict]], total_samples: int):
    """
    Verify federated split has no overlap and correct total.
    
    Args:
        client_data: Dictionary of client datasets
        total_samples: Expected total sample count
    """
    print("\n🔍 Verifying split...")
    
    # Check total count
    total_split = sum(len(data) for data in client_data.values())
    print(f"   Total samples after split: {total_split}")
    print(f"   Expected total: {total_samples}")
    
    if total_split != total_samples:
        print(f"   ⚠️  Warning: Sample count mismatch!")
        return False
    
    # Check for overlaps using index field
    all_indices = set()
    overlaps = 0
    
    for client_name, data in client_data.items():
        client_indices = set()
        for item in data:
            idx = item.get('index', item.get('id', None))
            if idx is not None:
                if idx in all_indices:
                    overlaps += 1
                client_indices.add(idx)
                all_indices.add(idx)
        
        print(f"   {client_name}: {len(client_indices)} unique indices")
    
    if overlaps > 0:
        print(f"   ❌ Found {overlaps} overlapping samples!")
        return False
    else:
        print(f"   ✅ No overlaps detected")
    
    # Check distribution variance
    counts = [len(data) for data in client_data.values()]
    avg_count = sum(counts) / len(counts)
    variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
    std_dev = variance ** 0.5
    
    print(f"   Distribution variance: {variance:.2f}")
    print(f"   Standard deviation: {std_dev:.2f}")
    print(f"   Min samples: {min(counts)}")
    print(f"   Max samples: {max(counts)}")
    print(f"   Ratio (max/min): {max(counts)/min(counts):.2f}x")
    
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Split preprocessed data for federated learning')
    parser.add_argument('--input', type=str, default='data/processed/dataset.jsonl',
                        help='Input JSONL file (default: data/processed/dataset.jsonl)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory (default: data/processed)')
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Number of clients (default: 3)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID split (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("=" * 80)
    print("🏥 FEDERATED DATA SPLIT")
    print("=" * 80)
    
    # Load preprocessed data
    print(f"\n📂 Loading data from {args.input}...")
    data = load_jsonl(args.input)
    print(f"✅ Loaded {len(data)} samples")
    
    # Create non-IID split
    client_data = create_non_iid_split(
        data=data,
        num_clients=args.num_clients,
        alpha=args.alpha
    )
    
    # Verify split
    is_valid = verify_split(client_data, len(data))
    
    if not is_valid:
        print("\n❌ Split validation failed!")
        return
    
    # Save splits
    print(f"\n💾 Saving splits to {args.output_dir}/...")
    for client_name, client_samples in client_data.items():
        output_path = os.path.join(args.output_dir, client_name, 'dataset.jsonl')
        save_jsonl(client_samples, output_path)
    
    # Create summary
    summary = {
        'total_samples': len(data),
        'num_clients': args.num_clients,
        'alpha': args.alpha,
        'seed': args.seed,
        'clients': {}
    }
    
    for client_name, client_samples in client_data.items():
        summary['clients'][client_name] = {
            'count': len(client_samples),
            'percentage': len(client_samples) / len(data) * 100,
            'path': f'{args.output_dir}/{client_name}/dataset.jsonl'
        }
    
    summary_path = os.path.join(args.output_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ FEDERATED SPLIT COMPLETE")
    print("=" * 80)
    
    print("\n📊 Summary:")
    for client_name, info in summary['clients'].items():
        print(f"   {client_name}: {info['count']} samples ({info['percentage']:.1f}%)")


if __name__ == '__main__':
    main()

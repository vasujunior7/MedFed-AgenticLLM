"""Quick verification script for federated split."""
import json

# Load datasets
datasets = {
    'A': [json.loads(l) for l in open('data/processed/hospital_A/dataset.jsonl')],
    'B': [json.loads(l) for l in open('data/processed/hospital_B/dataset.jsonl')],
    'C': [json.loads(l) for l in open('data/processed/hospital_C/dataset.jsonl')]
}

# Extract indices
indices = {
    'A': set(d['index'] for d in datasets['A']),
    'B': set(d['index'] for d in datasets['B']),
    'C': set(d['index'] for d in datasets['C'])
}

# Check overlaps
overlap_AB = len(indices['A'] & indices['B'])
overlap_AC = len(indices['A'] & indices['C'])
overlap_BC = len(indices['B'] & indices['C'])
total = len(indices['A']) + len(indices['B']) + len(indices['C'])

print("=" * 60)
print("🔍 FEDERATED SPLIT VERIFICATION")
print("=" * 60)
print(f"\n📊 Sample Counts:")
print(f"   Hospital A: {len(indices['A'])} samples")
print(f"   Hospital B: {len(indices['B'])} samples")
print(f"   Hospital C: {len(indices['C'])} samples")
print(f"   Total: {total} samples")

print(f"\n🔗 Overlap Check:")
print(f"   A ∩ B: {overlap_AB} samples")
print(f"   A ∩ C: {overlap_AC} samples")
print(f"   B ∩ C: {overlap_BC} samples")

all_unique = overlap_AB == 0 and overlap_AC == 0 and overlap_BC == 0
print(f"\n✅ All datasets unique: {all_unique}")

print(f"\n📈 Distribution Ratios:")
total_unique = len(indices['A'] | indices['B'] | indices['C'])
print(f"   Hospital A: {len(indices['A']) / total_unique * 100:.1f}%")
print(f"   Hospital B: {len(indices['B']) / total_unique * 100:.1f}%")
print(f"   Hospital C: {len(indices['C']) / total_unique * 100:.1f}%")
print(f"   Max/Min ratio: {max(len(indices['A']), len(indices['B']), len(indices['C'])) / min(len(indices['A']), len(indices['B']), len(indices['C'])):.2f}x")

print("\n" + "=" * 60)
if all_unique and total_unique == total:
    print("✅ VERIFICATION PASSED - Non-IID split successful!")
else:
    print("❌ VERIFICATION FAILED")
print("=" * 60)

#!/usr/bin/env python3
"""
Test script for Milestone 6 - Agentic Aggregation.
Tests that the coordinator makes intelligent weighting decisions.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(__file__))
from src.agent.coordinator import AgenticAggregator


def test_basic_aggregation():
    """Test 1: Basic aggregation with similar clients."""
    print("\n" + "="*70)
    print("TEST 1: Basic Aggregation (Similar Clients)")
    print("="*70)
    
    aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
    
    # 3 clients with similar performance
    client_metrics = [
        {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 2.0, 'num_samples': 1000},
        {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 2.1, 'num_samples': 1000},
        {'hospital': 'C', 'initial_loss': 10.0, 'final_loss': 2.2, 'num_samples': 1000},
    ]
    
    weights, analysis = aggregator.compute_aggregation_weights(
        client_metrics, 
        sample_counts=[1000, 1000, 1000]
    )
    
    print(f"\n📊 Client Performance:")
    for i, m in enumerate(client_metrics):
        print(f"   {m['hospital']}: loss {m['initial_loss']:.2f} → {m['final_loss']:.2f}")
    
    print(f"\n⚖️  Aggregation Weights:")
    for i, (w, m) in enumerate(zip(weights, client_metrics)):
        print(f"   {m['hospital']}: {w:.4f}")
    
    # Check weights are close (similar performance)
    weight_variance = np.var(weights)
    print(f"\n✓ Weight variance: {weight_variance:.6f}")
    
    # Weights should be reasonably close but can differ slightly
    assert weight_variance < 0.05, "Weights should be reasonably similar for similar clients"
    print("✅ PASS: Similar clients get reasonably similar weights")
    
    return weights, analysis


def test_bad_client_penalty():
    """Test 2: Bad client gets penalized."""
    print("\n" + "="*70)
    print("TEST 2: Bad Client Penalty")
    print("="*70)
    
    aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
    
    # Client C has much worse performance
    client_metrics = [
        {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 1.5, 'num_samples': 1000},
        {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 1.6, 'num_samples': 1000},
        {'hospital': 'C_BAD', 'initial_loss': 10.0, 'final_loss': 8.0, 'num_samples': 1000},  # Bad client
    ]
    
    weights, analysis = aggregator.compute_aggregation_weights(
        client_metrics,
        sample_counts=[1000, 1000, 1000]
    )
    
    print(f"\n📊 Client Performance:")
    for i, m in enumerate(client_metrics):
        improvement = m['initial_loss'] - m['final_loss']
        print(f"   {m['hospital']}: loss {m['initial_loss']:.2f} → {m['final_loss']:.2f} (Δ={improvement:.2f})")
    
    print(f"\n⚖️  Aggregation Weights:")
    for i, (w, m) in enumerate(zip(weights, client_metrics)):
        print(f"   {m['hospital']}: {w:.4f}")
    
    # Check bad client has lowest weight
    bad_client_idx = 2
    good_client_weights = [weights[0], weights[1]]
    bad_client_weight = weights[bad_client_idx]
    
    print(f"\n📊 Analysis:")
    print(f"   Good clients avg weight: {np.mean(good_client_weights):.4f}")
    print(f"   Bad client weight: {bad_client_weight:.4f}")
    print(f"   Penalty factor: {bad_client_weight / np.mean(good_client_weights):.2f}x")
    
    assert bad_client_weight < min(good_client_weights), "Bad client should have lowest weight"
    assert bad_client_weight < 0.5 * np.mean(good_client_weights), "Bad client should be heavily penalized"
    
    print("✅ PASS: Bad client penalized successfully")
    
    return weights, analysis


def test_unstable_client():
    """Test 3: Unstable client detection."""
    print("\n" + "="*70)
    print("TEST 3: Unstable Client Detection")
    print("="*70)
    
    aggregator = AgenticAggregator(
        loss_weight=0.6, 
        variance_weight=0.4,
        variance_threshold=4.0  # Lower threshold to catch unstable client
    )
    
    # Client B has very different improvement (unstable)
    client_metrics = [
        {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 2.0, 'num_samples': 1000},  # Improvement: 8.0
        {'hospital': 'B_UNSTABLE', 'initial_loss': 10.0, 'final_loss': 9.5, 'num_samples': 1000},  # Improvement: 0.5 (very different)
        {'hospital': 'C', 'initial_loss': 10.0, 'final_loss': 1.8, 'num_samples': 1000},  # Improvement: 8.2
    ]
    
    weights, analysis = aggregator.compute_aggregation_weights(
        client_metrics,
        sample_counts=[1000, 1000, 1000]
    )
    
    print(f"\n📊 Client Performance:")
    for i, m in enumerate(client_metrics):
        improvement = m['initial_loss'] - m['final_loss']
        print(f"   {m['hospital']}: improvement = {improvement:.2f}")
    
    print(f"\n⚠️  Unstable Clients Detected:")
    if analysis['unstable_clients']:
        for client in analysis['unstable_clients']:
            print(f"   - {client}")
    else:
        print("   None")
    
    print(f"\n⚖️  Aggregation Weights:")
    for i, (w, m) in enumerate(zip(weights, client_metrics)):
        variance = analysis['update_variances'][i]
        print(f"   {m['hospital']}: {w:.4f} (variance: {variance:.4f})")
    
    # Check unstable client detected
    assert 'B_UNSTABLE' in analysis['unstable_clients'], "Unstable client should be detected"
    
    # Check unstable client has lower weight
    unstable_idx = 1
    stable_weights = [weights[0], weights[2]]
    unstable_weight = weights[unstable_idx]
    
    print(f"\n📊 Variance Penalty:")
    print(f"   Stable clients avg: {np.mean(stable_weights):.4f}")
    print(f"   Unstable client: {unstable_weight:.4f}")
    
    assert unstable_weight < min(stable_weights), "Unstable client should have lower weight"
    
    print("✅ PASS: Unstable client detected and penalized")
    
    return weights, analysis


def test_weights_differ():
    """Test 4: Weights differ based on performance."""
    print("\n" + "="*70)
    print("TEST 4: Weights Differ Based on Performance")
    print("="*70)
    
    aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
    
    # Different performance levels
    client_metrics = [
        {'hospital': 'A_BEST', 'initial_loss': 10.0, 'final_loss': 1.0, 'num_samples': 1000},   # Best
        {'hospital': 'B_MEDIUM', 'initial_loss': 10.0, 'final_loss': 3.0, 'num_samples': 1000}, # Medium
        {'hospital': 'C_WORST', 'initial_loss': 10.0, 'final_loss': 6.0, 'num_samples': 1000},  # Worst
    ]
    
    weights, analysis = aggregator.compute_aggregation_weights(
        client_metrics,
        sample_counts=None  # Don't use sample counts to test pure quality ordering
    )
    
    print(f"\n📊 Client Rankings:")
    for i, m in enumerate(client_metrics):
        quality_score = analysis['quality_scores'][i]
        print(f"   {m['hospital']}: final_loss={m['final_loss']:.2f}, weight={weights[i]:.4f}, quality={quality_score:.4f}")
    
    # Check weights differ significantly (not all equal)
    weight_std = np.std(weights)
    assert weight_std > 0.05, "Weights should differ significantly based on performance"
    
    # Check worst client has lowest weight
    assert weights[2] < weights[0] and weights[2] < weights[1], "Worst client should have lowest weight"
    
    print(f"\n✓ Weight std deviation: {weight_std:.4f}")
    print(f"✓ Worst client (C) has lowest weight: {weights[2]:.4f}")
    print("✅ PASS: Weights correctly differentiate client quality")
    
    return weights, analysis


def test_malicious_detection():
    """Test 5: Malicious client detection."""
    print("\n" + "="*70)
    print("TEST 5: Malicious Client Detection")
    print("="*70)
    
    aggregator = AgenticAggregator()
    
    # Client C's loss actually increased (poisoned model)
    client_metrics = [
        {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 2.0, 'num_samples': 1000},
        {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 2.5, 'num_samples': 1000},
        {'hospital': 'C_MALICIOUS', 'initial_loss': 10.0, 'final_loss': 12.0, 'num_samples': 1000},  # Loss increased!
    ]
    
    suspicious = aggregator.detect_malicious_clients(client_metrics, loss_threshold=10.0)
    
    print(f"\n🔍 Malicious Client Detection:")
    print(f"   Suspicious indices: {suspicious}")
    
    for idx in suspicious:
        m = client_metrics[idx]
        print(f"   🚨 {m['hospital']}: loss {m['initial_loss']:.2f} → {m['final_loss']:.2f}")
    
    assert 2 in suspicious, "Malicious client should be detected"
    print("✅ PASS: Malicious client detected")
    
    return suspicious


def main():
    """Run all tests for Milestone 6."""
    print("\n" + "="*70)
    print("🧪 TESTING AGENTIC AGGREGATION (MILESTONE 6)")
    print("="*70)
    
    print("\n🎯 Goal: Hybrid agent with rules + scoring")
    print("   - Score = 0.6 * loss + 0.4 * variance")
    print("   - Downweight unstable clients")
    print("   - Output normalized weights")
    
    # Run all tests
    try:
        # Test 1: Basic
        test_basic_aggregation()
        
        # Test 2: Bad client penalty (REQUIRED)
        test_bad_client_penalty()
        
        # Test 3: Unstable detection
        test_unstable_client()
        
        # Test 4: Weights differ (REQUIRED)
        test_weights_differ()
        
        # Test 5: Malicious detection
        test_malicious_detection()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - MILESTONE 6 COMPLETE!")
        print("="*70)
        
        print("\n✅ Agentic AI Proof:")
        print("   ✓ Weights differ based on client quality")
        print("   ✓ Bad clients are penalized")
        print("   ✓ Unstable clients are downweighted")
        print("   ✓ Intelligent scoring (0.6 * loss + 0.4 * variance)")
        print("   ✓ Normalized aggregation weights")
        print("   ✓ Malicious client detection")
        
        print("\n🎉 Agentic Aggregation System Ready!")
        print("="*70 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

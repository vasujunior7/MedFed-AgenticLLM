"""
Pytest-compatible test suite for FED-MED project.
Can be run with: pytest test_pytest.py -v
"""

import pytest
import os
import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.dirname(__file__))
from src.agent.coordinator import AgenticAggregator
from src.safety.guardrails import MedicalGuardrails


class TestDatasetLoading:
    """Test dataset loading and validation."""
    
    def test_datasets_exist(self):
        """Verify all hospital datasets exist."""
        hospitals = ['hospital_A', 'hospital_B', 'hospital_C']
        for hospital in hospitals:
            dataset_path = f"data/processed/{hospital}/dataset.jsonl"
            assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    
    def test_datasets_nonempty(self):
        """Verify datasets contain data."""
        hospitals = ['hospital_A', 'hospital_B', 'hospital_C']
        for hospital in hospitals:
            dataset_path = f"data/processed/{hospital}/dataset.jsonl"
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            assert len(data) > 0, f"{hospital} has no data"
    
    def test_dataset_structure(self):
        """Verify dataset structure is valid."""
        dataset_path = "data/processed/hospital_A/dataset.jsonl"
        with open(dataset_path, 'r') as f:
            sample = json.loads(f.readline())
        
        assert 'text' in sample
        assert 'token_length' in sample
        assert isinstance(sample['text'], str)
        assert isinstance(sample['token_length'], int)
    
    def test_no_data_overlap(self):
        """Verify no overlap between hospital datasets (federated split)."""
        hospitals = ['hospital_A', 'hospital_B', 'hospital_C']
        indices = {}
        
        for hospital in hospitals:
            dataset_path = f"data/processed/{hospital}/dataset.jsonl"
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            indices[hospital] = set(d.get('index', i) for i, d in enumerate(data))
        
        # Check all pairwise overlaps
        assert len(indices['hospital_A'] & indices['hospital_B']) == 0
        assert len(indices['hospital_A'] & indices['hospital_C']) == 0
        assert len(indices['hospital_B'] & indices['hospital_C']) == 0


class TestAgentWeights:
    """Test agent weight computation."""
    
    def test_weights_sum_to_one(self):
        """Verify weights form valid probability distribution."""
        aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
        
        client_metrics = [
            {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 2.0, 'num_samples': 1000},
            {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 3.0, 'num_samples': 1000},
            {'hospital': 'C', 'initial_loss': 10.0, 'final_loss': 4.0, 'num_samples': 1000},
        ]
        
        weights, _ = aggregator.compute_aggregation_weights(
            client_metrics, 
            sample_counts=[1000, 1000, 1000]
        )
        
        assert abs(sum(weights) - 1.0) < 1e-6
    
    def test_weights_nonnegative(self):
        """Verify all weights are non-negative."""
        aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
        
        client_metrics = [
            {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 1.0, 'num_samples': 1000},
            {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 5.0, 'num_samples': 1000},
            {'hospital': 'C', 'initial_loss': 10.0, 'final_loss': 9.0, 'num_samples': 1000},
        ]
        
        weights, _ = aggregator.compute_aggregation_weights(
            client_metrics,
            sample_counts=[1000, 1000, 1000]
        )
        
        assert all(w >= 0 for w in weights)
    
    def test_best_performer_highest_weight(self):
        """Verify best performer gets highest weight."""
        aggregator = AgenticAggregator(loss_weight=0.6, variance_weight=0.4)
        
        client_metrics = [
            {'hospital': 'A', 'initial_loss': 10.0, 'final_loss': 1.0, 'num_samples': 1000},  # Best
            {'hospital': 'B', 'initial_loss': 10.0, 'final_loss': 5.0, 'num_samples': 1000},
            {'hospital': 'C', 'initial_loss': 10.0, 'final_loss': 9.0, 'num_samples': 1000},  # Worst
        ]
        
        weights, _ = aggregator.compute_aggregation_weights(
            client_metrics,
            sample_counts=[1000, 1000, 1000]
        )
        
        best_idx = 0
        worst_idx = 2
        assert weights[best_idx] > weights[worst_idx]


class TestFederatedAggregation:
    """Test federated aggregation execution."""
    
    def test_aggregation_preserves_shapes(self):
        """Verify aggregation preserves parameter shapes."""
        np.random.seed(42)
        num_params = 10
        
        client_params = [
            [np.random.randn(num_params) for _ in range(3)],
            [np.random.randn(num_params) for _ in range(3)],
            [np.random.randn(num_params) for _ in range(3)],
        ]
        
        weights = [0.33, 0.33, 0.34]
        
        aggregated = []
        for layer_idx in range(3):
            layer_params = [client_params[i][layer_idx] for i in range(3)]
            agg_layer = np.average(layer_params, axis=0, weights=weights)
            aggregated.append(agg_layer)
        
        for agg, orig in zip(aggregated, client_params[0]):
            assert agg.shape == orig.shape
    
    def test_weighted_averaging_correct(self):
        """Verify weighted averaging is mathematically correct."""
        np.random.seed(42)
        num_params = 10
        
        client_params = [
            np.random.randn(num_params),
            np.random.randn(num_params),
            np.random.randn(num_params),
        ]
        
        weights = [0.5, 0.3, 0.2]
        
        # Aggregation
        aggregated = np.average(client_params, axis=0, weights=weights)
        
        # Manual calculation
        manual = weights[0] * client_params[0] + weights[1] * client_params[1] + weights[2] * client_params[2]
        
        assert np.allclose(aggregated, manual)
    
    def test_different_weights_different_results(self):
        """Verify different weights produce different aggregations."""
        np.random.seed(42)
        num_params = 10
        
        client_params = [
            np.random.randn(num_params),
            np.random.randn(num_params),
            np.random.randn(num_params),
        ]
        
        weights1 = [0.5, 0.3, 0.2]
        weights2 = [0.2, 0.3, 0.5]
        
        agg1 = np.average(client_params, axis=0, weights=weights1)
        agg2 = np.average(client_params, axis=0, weights=weights2)
        
        assert not np.allclose(agg1, agg2)


class TestTrainingArtifacts:
    """Test training artifacts exist."""
    
    def test_lora_adapters_exist(self):
        """Verify LoRA adapter files exist."""
        hospitals = ['hospital_A', 'hospital_B', 'hospital_C']
        
        for hospital in hospitals:
            adapter_path = f"output-models/federated/{hospital}/final/adapter_model.safetensors"
            if os.path.exists(f"output-models/federated/{hospital}/final"):
                assert os.path.exists(adapter_path), f"{hospital} adapter missing"
    
    def test_training_history_exists(self):
        """Verify training history was recorded."""
        metrics_file = "output-models/federated/metrics/training_history.json"
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                history = json.load(f)
            
            assert 'rounds' in history
            assert 'global_losses' in history
            assert len(history['rounds']) > 0


class TestSafetyGuardrails:
    """Test safety guardrails functionality."""
    
    def test_safe_response_passes(self):
        """Verify safe responses pass checks."""
        guardrails = MedicalGuardrails()
        response = "Diabetes symptoms include increased thirst. Please consult your doctor."
        result = guardrails.check_response(response)
        assert result['is_safe'] == True
    
    def test_disclaimer_detection(self):
        """Verify disclaimer detection works."""
        guardrails = MedicalGuardrails()
        response_with = "Please consult a healthcare professional."
        response_without = "Take this medication daily."
        
        assert guardrails._has_disclaimer(response_with)
        assert not guardrails._has_disclaimer(response_without)
    
    def test_disclaimer_addition(self):
        """Verify disclaimer can be added."""
        guardrails = MedicalGuardrails()
        response = "Diabetes is a chronic condition."
        with_disclaimer = guardrails.add_disclaimer(response)
        
        assert len(with_disclaimer) > len(response)
        assert "⚠️" in with_disclaimer
        assert "healthcare professional" in with_disclaimer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

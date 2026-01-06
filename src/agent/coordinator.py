"""Agentic coordinator for intelligent federated aggregation."""

from typing import List, Dict, Tuple
import numpy as np
import logging


class AgenticAggregator:
    """
    Intelligent aggregation coordinator with client scoring.
    
    This is the AGENTIC component that makes intelligent decisions
    about which clients to trust more based on their training quality.
    """
    
    def __init__(
        self, 
        loss_weight: float = 0.6,
        variance_weight: float = 0.4,
        min_weight: float = 0.1,
        variance_threshold: float = 5.0
    ):
        """
        Initialize agentic aggregator.
        
        Args:
            loss_weight: Weight for loss component in score (0-1)
            variance_weight: Weight for variance component in score (0-1)
            min_weight: Minimum weight for any client (prevents complete exclusion)
            variance_threshold: Threshold above which clients are heavily penalized
        """
        self.loss_weight = loss_weight
        self.variance_weight = variance_weight
        self.min_weight = min_weight
        self.variance_threshold = variance_threshold
        self.logger = logging.getLogger(__name__)
        
        assert abs(loss_weight + variance_weight - 1.0) < 1e-6, "Weights must sum to 1"
    
    def compute_client_scores(
        self, 
        client_metrics: List[Dict]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute quality scores for each client based on training metrics.
        
        AGENTIC LOGIC:
        - score = 0.6 * loss_component + 0.4 * variance_component
        - Lower loss is better (normalized)
        - Lower variance is better (more stable training)
        - Bad clients get penalized
        
        Args:
            client_metrics: List of metric dicts from each client
                Each dict should have:
                - 'initial_loss': float
                - 'final_loss': float
                - 'hospital': str
                - 'num_samples': int
        
        Returns:
            Tuple of (scores array, analysis dict)
        """
        if not client_metrics:
            raise ValueError("No client metrics provided")
        
        num_clients = len(client_metrics)
        scores = np.zeros(num_clients)
        
        # Extract metrics
        final_losses = np.array([m['final_loss'] for m in client_metrics])
        initial_losses = np.array([m['initial_loss'] for m in client_metrics])
        
        # Compute loss improvement (larger is better)
        loss_improvements = initial_losses - final_losses
        
        # Compute update variance (measure of training stability)
        # Deviation from mean improvement indicates instability
        mean_improvement = np.mean(loss_improvements)
        update_variances = np.abs(loss_improvements - mean_improvement)
        
        # Normalize to [0, 1] range
        # For loss: lower final loss is better (inverse normalize)
        if final_losses.max() > final_losses.min():
            loss_component = 1 - (final_losses - final_losses.min()) / (final_losses.max() - final_losses.min())
        else:
            loss_component = np.ones(num_clients)
        
        # For variance: lower variance is better (inverse normalize)
        max_var = update_variances.max()
        if max_var > 1e-6:  # Only normalize if there's meaningful variance
            variance_component = 1 - (update_variances / max_var)
        else:
            variance_component = np.ones(num_clients)
        
        # Apply variance penalty for unstable clients
        # Only penalize if variance is significantly high
        high_variance_mask = update_variances > self.variance_threshold
        if high_variance_mask.any():
            variance_component[high_variance_mask] *= 0.5  # Heavy penalty
        
        # Compute final scores (weighted combination)
        scores = (self.loss_weight * loss_component + 
                 self.variance_weight * variance_component)
        
        # Analysis for logging
        analysis = {
            'final_losses': final_losses.tolist(),
            'loss_improvements': loss_improvements.tolist(),
            'update_variances': update_variances.tolist(),
            'loss_component': loss_component.tolist(),
            'variance_component': variance_component.tolist(),
            'raw_scores': scores.tolist(),
            'unstable_clients': [client_metrics[i]['hospital'] 
                               for i in range(num_clients) if high_variance_mask[i]]
        }
        
        return scores, analysis
    
    def compute_aggregation_weights(
        self,
        client_metrics: List[Dict],
        sample_counts: List[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute normalized aggregation weights for federated averaging.
        
        AGENTIC DECISION: Combines quality scores with sample counts
        to determine how much to trust each client's update.
        
        Args:
            client_metrics: List of metric dicts from each client
            sample_counts: Optional list of sample counts per client
        
        Returns:
            Tuple of (normalized weights array, detailed analysis)
        """
        num_clients = len(client_metrics)
        
        # Compute quality scores (agentic component)
        quality_scores, analysis = self.compute_client_scores(client_metrics)
        
        # Apply minimum weight threshold
        quality_scores = np.maximum(quality_scores, self.min_weight)
        
        # If sample counts provided, combine with quality scores
        if sample_counts is not None:
            sample_weights = np.array(sample_counts, dtype=float)
            sample_weights = sample_weights / sample_weights.sum()
            
            # Combine: 70% quality, 30% sample count
            combined_weights = 0.7 * quality_scores + 0.3 * sample_weights
        else:
            combined_weights = quality_scores
        
        # Normalize to sum to 1
        weights = combined_weights / combined_weights.sum()
        
        # Add to analysis
        analysis['quality_scores'] = quality_scores.tolist()
        analysis['final_weights'] = weights.tolist()
        if sample_counts is not None:
            analysis['sample_weights'] = sample_weights.tolist()
        
        # Log decisions
        self.logger.info(f"📊 Agentic Aggregation Weights:")
        for i, (w, m) in enumerate(zip(weights, client_metrics)):
            self.logger.info(f"   {m['hospital']}: {w:.4f} (quality: {quality_scores[i]:.4f})")
        
        if analysis['unstable_clients']:
            self.logger.warning(f"⚠️  Unstable clients detected: {analysis['unstable_clients']}")
        
        return weights, analysis
    
    def detect_malicious_clients(
        self,
        client_metrics: List[Dict],
        loss_threshold: float = 10.0
    ) -> List[int]:
        """
        Detect potentially malicious or failing clients.
        
        Args:
            client_metrics: List of metric dicts
            loss_threshold: Final loss threshold for flagging
        
        Returns:
            List of client indices that are suspicious
        """
        suspicious = []
        
        for i, metrics in enumerate(client_metrics):
            # Flag if final loss is too high
            if metrics['final_loss'] > loss_threshold:
                suspicious.append(i)
                self.logger.warning(
                    f"🚨 Client {metrics['hospital']} flagged: "
                    f"high final loss ({metrics['final_loss']:.4f})"
                )
            
            # Flag if loss increased (model got worse)
            if metrics['final_loss'] > metrics['initial_loss']:
                suspicious.append(i)
                self.logger.warning(
                    f"🚨 Client {metrics['hospital']} flagged: "
                    f"loss increased during training"
                )
        
        return list(set(suspicious))  # Remove duplicates


class FederatedCoordinator:
    """Coordinate federated learning across multiple hospitals."""
    
    def __init__(self, config: Dict):
        """
        Initialize coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.aggregator = AgenticAggregator()
    
    def initialize_clients(self) -> List:
        """Initialize federated learning clients."""
        # TODO: Implement client initialization
        pass
    
    def start_training_round(self, round_num: int):
        """
        Start a training round.
        
        Args:
            round_num: Current round number
        """
        # TODO: Implement training round
        pass
    
    def aggregate_results(self):
        """Aggregate results from all clients."""
        # TODO: Implement result aggregation
        pass

#!/usr/bin/env python3
"""
Shared State Manager for FED-MED Gradio App

This module provides a thread-safe shared state system that allows
the federated training backend to communicate with the Gradio frontend
in near real-time.

State includes:
- Current federated round
- Active hospitals
- Agent aggregation weights
- Agent decisions (trusted/penalized clients)
- Training metrics
- Architecture visualization state
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class SharedState:
    """Thread-safe shared state manager for FED-MED system."""
    
    def __init__(self, state_file: str = "shared_state.json"):
        """
        Initialize shared state manager.
        
        Args:
            state_file: Path to JSON file storing state
        """
        self.state_file = Path(state_file)
        self.lock = threading.Lock()
        
        # Initialize default state
        self.default_state = {
            "federated": {
                "current_round": 0,
                "total_rounds": 3,
                "status": "idle",  # idle, training, aggregating, complete
                "global_loss": None,
            },
            "hospitals": {
                "hospital_A": {
                    "name": "Hospital A",
                    "samples": 4520,
                    "status": "idle",  # idle, training, completed
                    "loss": None,
                    "last_update": None
                },
                "hospital_B": {
                    "name": "Hospital B",
                    "samples": 2521,
                    "status": "idle",
                    "loss": None,
                    "last_update": None
                },
                "hospital_C": {
                    "name": "Hospital C",
                    "samples": 2959,
                    "status": "idle",
                    "loss": None,
                    "last_update": None
                }
            },
            "agent": {
                "weights": {
                    "hospital_A": 0.33,
                    "hospital_B": 0.33,
                    "hospital_C": 0.33
                },
                "decisions": {
                    "trusted": [],
                    "penalized": [],
                    "unstable": []
                },
                "last_decision": None,
                "aggregation_method": "agentic"
            },
            "training_history": {
                "rounds": [],
                "losses": [],
                "timestamps": []
            },
            "inference": {
                "status": "not_loaded",  # not_loaded, loading, ready
                "model_name": "Hospital B (Best Performer)",
                "queries_processed": 0
            },
            "architecture_viz": {
                "active_flows": [],  # List of active data flows for animation
                "highlight_node": None  # Currently highlighted node
            },
            "metadata": {
                "last_update": None,
                "version": "1.0.0"
            }
        }
        
        # Load or initialize state
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load state from file or initialize with defaults."""
        with self.lock:
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        self.state = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load state file: {e}")
                    self.state = self.default_state.copy()
            else:
                self.state = self.default_state.copy()
            
            self._save_unsafe()
    
    def _save_unsafe(self):
        """Save state to file without acquiring lock (internal use)."""
        self.state["metadata"]["last_update"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get(self, key: Optional[str] = None) -> Any:
        """
        Get state value.
        
        Args:
            key: Dot-separated path (e.g., "federated.current_round")
                 If None, returns entire state
        
        Returns:
            State value or entire state dict
        """
        with self.lock:
            if key is None:
                return self.state.copy()
            
            # Navigate nested dict
            keys = key.split('.')
            value = self.state
            for k in keys:
                value = value.get(k, {})
            return value
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Set state value.
        
        Args:
            key: Dot-separated path (e.g., "federated.current_round")
            value: Value to set
            save: Whether to save to disk immediately
        """
        with self.lock:
            # Navigate and set
            keys = key.split('.')
            target = self.state
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value
            
            if save:
                self._save_unsafe()
    
    def update_federated_round(self, round_num: int, status: str = "training"):
        """Update current federated round."""
        with self.lock:
            self.state["federated"]["current_round"] = round_num
            self.state["federated"]["status"] = status
            self._save_unsafe()
    
    def update_hospital_status(self, hospital: str, status: str, loss: Optional[float] = None):
        """Update hospital training status."""
        with self.lock:
            if hospital in self.state["hospitals"]:
                self.state["hospitals"][hospital]["status"] = status
                if loss is not None:
                    self.state["hospitals"][hospital]["loss"] = loss
                self.state["hospitals"][hospital]["last_update"] = datetime.now().isoformat()
                self._save_unsafe()
    
    def update_agent_weights(self, weights: Dict[str, float], decisions: Optional[Dict] = None):
        """Update agent aggregation weights."""
        with self.lock:
            self.state["agent"]["weights"] = weights
            if decisions:
                self.state["agent"]["decisions"] = decisions
            self.state["agent"]["last_decision"] = datetime.now().isoformat()
            self._save_unsafe()
    
    def update_global_loss(self, loss: float):
        """Update global model loss."""
        with self.lock:
            self.state["federated"]["global_loss"] = loss
            self._save_unsafe()
    
    def add_training_record(self, round_num: int, loss: float):
        """Add training history record."""
        with self.lock:
            self.state["training_history"]["rounds"].append(round_num)
            self.state["training_history"]["losses"].append(loss)
            self.state["training_history"]["timestamps"].append(datetime.now().isoformat())
            self._save_unsafe()
    
    def update_inference_status(self, status: str, queries_processed: Optional[int] = None):
        """Update inference engine status."""
        with self.lock:
            self.state["inference"]["status"] = status
            if queries_processed is not None:
                self.state["inference"]["queries_processed"] = queries_processed
            self._save_unsafe()
    
    def increment_queries(self):
        """Increment query counter."""
        with self.lock:
            self.state["inference"]["queries_processed"] += 1
            self._save_unsafe()
    
    def add_active_flow(self, from_node: str, to_node: str):
        """Add active data flow for visualization."""
        with self.lock:
            flow = {"from": from_node, "to": to_node, "timestamp": datetime.now().isoformat()}
            self.state["architecture_viz"]["active_flows"].append(flow)
            # Keep only last 5 flows
            self.state["architecture_viz"]["active_flows"] = \
                self.state["architecture_viz"]["active_flows"][-5:]
            self._save_unsafe()
    
    def highlight_node(self, node: Optional[str]):
        """Highlight a node in architecture visualization."""
        with self.lock:
            self.state["architecture_viz"]["highlight_node"] = node
            self._save_unsafe()
    
    def reset(self):
        """Reset state to defaults."""
        with self.lock:
            self.state = self.default_state.copy()
            self._save_unsafe()
    
    def get_summary(self) -> str:
        """Get human-readable state summary."""
        state = self.get()
        
        summary = f"""
FED-MED System State
{'='*50}

Federated Training:
  Round: {state['federated']['current_round']}/{state['federated']['total_rounds']}
  Status: {state['federated']['status']}
  Global Loss: {state['federated']['global_loss']}

Hospitals:
"""
        for hospital_id, info in state['hospitals'].items():
            summary += f"  {info['name']}: {info['status']} (Loss: {info['loss']})\n"
        
        summary += f"""
Agent Decisions:
  Aggregation: {state['agent']['aggregation_method']}
  Weights: {state['agent']['weights']}
  Trusted: {state['agent']['decisions']['trusted']}
  Penalized: {state['agent']['decisions']['penalized']}

Inference:
  Status: {state['inference']['status']}
  Queries Processed: {state['inference']['queries_processed']}
"""
        return summary


# Global singleton instance
_global_state = None

def get_shared_state() -> SharedState:
    """Get global shared state instance."""
    global _global_state
    if _global_state is None:
        _global_state = SharedState()
    return _global_state


if __name__ == "__main__":
    # Test shared state
    state = get_shared_state()
    
    print("Initial State:")
    print(state.get_summary())
    
    # Simulate updates
    state.update_federated_round(1, "training")
    state.update_hospital_status("hospital_A", "training", 0.35)
    state.update_agent_weights({
        "hospital_A": 0.25,
        "hospital_B": 0.55,
        "hospital_C": 0.20
    })
    
    print("\nUpdated State:")
    print(state.get_summary())

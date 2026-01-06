"""Federated learning server implementation."""

import flwr as fl
from typing import Dict, Optional, Tuple
import yaml


def start_federated_server(config_path: str = "src/config/federated_config.yaml"):
    """
    Start federated learning server.
    
    Args:
        config_path: Path to federated configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config['federated']['fraction_fit'],
        fraction_evaluate=config['federated']['fraction_evaluate'],
        min_available_clients=config['federated']['min_available_clients']
    )
    
    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=config['federated']['num_rounds']),
        strategy=strategy
    )


if __name__ == "__main__":
    start_federated_server()

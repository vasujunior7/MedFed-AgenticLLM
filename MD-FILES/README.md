# FED-MED: Federated Medical AI with LoRA Fine-tuning

A federated learning framework for training medical AI assistants across distributed hospital data while preserving privacy.

## 🏥 Overview

This project implements federated learning for medical question-answering using:
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Federated Learning** across 4 hospital clients
- **Safety Guardrails** for medical AI responses
- **32GB VRAM optimized** training configuration

## 📁 Project Structure

```
FED-MED/
├── data/                    # Dataset storage
│   ├── raw/                 # Original medical dataset
│   └── processed/           # Federated splits per hospital
├── notebooks/               # Jupyter notebooks for experimentation
├── src/                     # Source code
│   ├── config/             # YAML configurations
│   ├── data/               # Data loading and preprocessing
│   ├── model/              # Model loading and LoRA setup
│   ├── federated/          # Federated learning components
│   ├── agent/              # Coordination and metrics
│   ├── training/           # Training utilities
│   ├── safety/             # Medical safety guardrails
│   └── utils/              # Logging and utilities
├── tests/                   # Unit tests
└── demo/                    # Demo application
```

## 🚀 Configuration Highlights

### LoRA Configuration (32GB VRAM Optimized)
- `lora_r: 16` - Stronger adaptation capacity
- `lora_alpha: 32` - Increased scaling
- Target modules: `q_proj`, `v_proj`, `k_proj`
- ~1-3% trainable parameters

### Training Configuration
- Batch size: 2 with 4 gradient accumulation steps
- Effective batch size: 8
- Learning rate: 1e-4 (stable)
- 2 epochs on 30k samples
- BF16 precision for 32GB GPUs

### Federated Learning
- 4 hospital clients (A, B, C, D)
- FedAvg aggregation
- HIPAA/GDPR compliance ready
- Privacy-preserving training

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (if needed)
python src/data/load_dataset.py
```

## 🎯 Quick Start

### 1. Data Preparation
```python
from src.data.load_dataset import load_medical_dataset
from src.data.federated_split import create_federated_split

# Load dataset
dataset = load_medical_dataset()

# Split across hospitals
client_datasets = create_federated_split(dataset, num_clients=4)
```

### 2. Local Training (Single Hospital)
```python
from src.model.load_model import load_model_and_tokenizer
from src.model.lora_setup import setup_lora
from src.training.local_train import train_local

# Load and setup model
model, tokenizer = load_model_and_tokenizer()
model = setup_lora(model)

# Train locally
trainer = train_local(model, train_dataset)
```

### 3. Federated Training
```python
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient

# Initialize server and clients
server = FederatedServer(initial_model)
clients = [FederatedClient(f"hospital_{i}", data) 
           for i, data in enumerate(client_datasets.values())]

# Run federated rounds
for round in range(10):
    # Distribute global model
    global_state = server.get_global_model_state()
    
    # Local training
    updates = [client.local_training() for client in clients]
    weights = [client.get_num_samples() for client in clients]
    
    # Aggregate and update
    aggregated = server.aggregate_updates(updates, weights)
    server.update_global_model(aggregated)
```

### 4. Inference with Safety
```python
from src.model.inference import generate_response
from src.safety.guardrails import MedicalGuardrails

# Initialize guardrails
guardrails = MedicalGuardrails()

# Generate response
response = generate_response(model, tokenizer, "What causes diabetes?")

# Safety check
is_safe, reason = guardrails.check_response(response)
if is_safe:
    response = guardrails.add_disclaimer(response)
    print(response)
```

## 📊 Monitoring

Training metrics are logged to:
- TensorBoard: `logs/`
- Metrics tracker: `logs/metrics/`
- Model checkpoints: `output-models/`

```bash
# View TensorBoard
tensorboard --logdir=logs
```

## 🔒 Safety Features

- **Response validation**: Checks for overconfident or dangerous patterns
- **Medical disclaimers**: Auto-added to responses
- **Confidence thresholds**: Min 0.3, Max 0.95
- **HIPAA compliance**: Privacy-preserving federated learning

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Specific test
pytest tests/test_federated.py
```

## 📝 Configuration Files

All configurations are in `src/config/`:
- `model_config.yaml` - LoRA and model settings
- `training_config.yaml` - Training hyperparameters  
- `federated_config.yaml` - Federated learning setup

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure safety guardrails are maintained

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Dataset: AI-MO/ai-medical-dataset
- Framework: HuggingFace Transformers, PEFT
- Federated Learning: Based on FedAvg algorithm

---

**⚠️ Important**: This is a research/educational project. All medical outputs should be reviewed by qualified healthcare professionals before any real-world use.

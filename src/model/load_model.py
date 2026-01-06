"""Load and initialize language models."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml


def load_model_and_tokenizer(config_path: str = "src/config/model_config.yaml"):
    """
    Load model and tokenizer based on configuration.
    
    Args:
        config_path: Path to model configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

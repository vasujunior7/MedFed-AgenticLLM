"""Setup LoRA for parameter-efficient fine-tuning."""

from peft import LoraConfig, get_peft_model, TaskType
import yaml


def setup_lora(model, config_path: str = "src/config/model_config.yaml"):
    """
    Setup LoRA configuration and apply to model.
    
    Args:
        model: Base model
        config_path: Path to model configuration
        
    Returns:
        Model with LoRA adapters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        task_type=TaskType.CAUSAL_LM
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model

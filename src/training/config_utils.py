"""
Configuration utilities for loading training configs from YAML files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage training configurations"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize config loader
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML configuration file
        
        Args:
            filename: Name of YAML file (with or without .yaml extension)
            
        Returns:
            Dictionary with config
        """
        
        if not filename.endswith(".yaml"):
            filename += ".yaml"
        
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Loading config: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.configs[filename] = config
        return config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.load_yaml("training_config.yaml")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.load_yaml("model_config.yaml")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.load_yaml("data_config.yaml")
    
    def get_main_config(self) -> Dict[str, Any]:
        """Get main project configuration"""
        return self.load_yaml("config.yaml")
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all standard configs"""
        configs = {}
        configs["main"] = self.get_main_config()
        configs["model"] = self.get_model_config()
        configs["training"] = self.get_training_config()
        configs["data"] = self.get_data_config()
        
        return configs
    
    @staticmethod
    def create_training_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config dict to TrainingArguments kwargs
        
        Args:
            config: Training config dictionary
            
        Returns:
            Dictionary with TrainingArguments parameters
        """
        
        training_config = config.get("training", {})
        
        return {
            "output_dir": training_config.get("output_dir", "outputs"),
            "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
            "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 1),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 4),
            "warmup_steps": training_config.get("warmup_steps", 10),
            "num_train_epochs": training_config.get("num_train_epochs", 3),
            "learning_rate": training_config.get("learning_rate", 1e-5),
            "weight_decay": training_config.get("weight_decay", 0.01),
            "bf16": training_config.get("bf16", True),
            "fp16": training_config.get("fp16", False),
            "optim": training_config.get("optim", "adamw_8bit"),
            "max_grad_norm": training_config.get("max_grad_norm", 1.0),
            "logging_steps": training_config.get("logging_steps", 1),
            "eval_steps": training_config.get("eval_steps", 50),
            "save_steps": training_config.get("save_steps", 200),
            "save_strategy": training_config.get("save_strategy", "steps"),
            "save_total_limit": training_config.get("save_total_limit", 2),
            "logging_strategy": training_config.get("logging_strategy", "steps"),
            "evaluation_strategy": training_config.get("evaluation_strategy", "steps"),
            "remove_unused_columns": training_config.get("remove_unused_columns", False),
            "seed": training_config.get("seed", 42),
            "data_seed": training_config.get("data_seed", 42),
            "report_to": training_config.get("report_to", []),
        }
    
    @staticmethod
    def create_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config dict to LoRA config
        
        Args:
            config: Model config dictionary
            
        Returns:
            Dictionary with LoRA parameters
        """
        
        lora_config = config.get("lora", {})
        
        return {
            "r": lora_config.get("r", 32),
            "lora_alpha": lora_config.get("lora_alpha", 32),
            "lora_dropout": lora_config.get("lora_dropout", 0),
            "bias": lora_config.get("bias", "none"),
            "target_modules": lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            "use_gradient_checkpointing": lora_config.get("use_gradient_checkpointing", "unsloth"),
            "use_rslora": lora_config.get("use_rslora", True),
            "random_state": lora_config.get("random_state", 3407),
            "loftq_config": lora_config.get("loftq_config", None),
        }
    
    @staticmethod
    def setup_environment(config: Dict[str, Any]) -> None:
        """
        Setup environment variables from config
        
        Args:
            config: Configuration dictionary with 'environment' section
        """
        
        env_config = config.get("environment", {})
        
        if env_config.get("pytorch_cuda_alloc_conf"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_config["pytorch_cuda_alloc_conf"]
        
        if env_config.get("hf_hub_connect_timeout"):
            os.environ["HF_HUB_CONNECT_TIMEOUT"] = str(env_config["hf_hub_connect_timeout"])
        
        if env_config.get("hf_hub_read_timeout"):
            os.environ["HF_HUB_READ_TIMEOUT"] = str(env_config["hf_hub_read_timeout"])
        
        if env_config.get("hf_hub_etag_timeout"):
            os.environ["HF_HUB_ETAG_TIMEOUT"] = str(env_config["hf_hub_etag_timeout"])
        
        if env_config.get("hf_hub_disable_telemetry"):
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        print("Environment variables set up")


if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader()
    
    # Load all configs
    all_configs = loader.get_all_configs()
    
    # Get training args
    training_config = loader.get_training_config()
    training_args = ConfigLoader.create_training_args(training_config)
    
    print("Training Arguments:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Get LoRA config
    model_config = loader.get_model_config()
    lora_config = ConfigLoader.create_lora_config(model_config)
    
    print("\nLoRA Configuration:")
    for key, value in lora_config.items():
        print(f"  {key}: {value}")

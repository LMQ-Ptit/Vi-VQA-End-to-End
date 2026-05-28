"""
Model loading utilities for Vi-VQA
Supports loading 7B teacher and 2B student models from HuggingFace
"""

import os
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False


class ModelConfig:
    """Model configuration container"""
    
    def __init__(self, model_id: str, model_name: str, params: str, purpose: str):
        self.model_id = model_id
        self.model_name = model_name
        self.params = params
        self.purpose = purpose


# Pre-configured models
MODELS = {
    "7B-base": ModelConfig(
        model_id="7B-base",
        model_name="unsloth/Qwen2-VL-7B-Instruct",
        params="7B",
        purpose="base_model_for_training",
    ),
    "7B-trained": ModelConfig(
        model_id="7B-trained",
        model_name="MinhQuy24/Qwen-7B-Vi-VQA",
        params="7B",
        purpose="teacher_model",
    ),
    "2B-distilled": ModelConfig(
        model_id="2B-distilled",
        model_name="MinhQuy24/Qwen-2B-ViVQA",
        params="2B",
        purpose="student_model_distilled",
    ),
}


class VQAModelLoader:
    """Load and manage VQA models from HuggingFace Hub"""
    
    def __init__(self, device: str = "cuda", use_auth_token: bool = True):
        """
        Initialize model loader
        
        Args:
            device: Device to load models on ("cuda" or "cpu")
            use_auth_token: Use HuggingFace authentication token
        """
        self.device = device
        self.use_auth_token = use_auth_token
        self.models = {}
        self.tokenizers = {}
        
    def load_model(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        max_memory: Optional[dict] = None,
    ) -> Tuple[torch.nn.Module, object]:
        """
        Load model and tokenizer from HuggingFace
        
        Args:
            model_name: Model name or path
            load_in_4bit: Load in 4-bit quantization
            load_in_8bit: Load in 8-bit quantization
            device_map: Device mapping strategy
            max_memory: Max memory per device
            
        Returns:
            Tuple of (model, tokenizer)
        """
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=self.use_auth_token,
            trust_remote_code=True,
        )
        
        # Load model
        print(f"  Loading model...")
        
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            use_auth_token=self.use_auth_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        )
        
        print(f"  Model loaded successfully!")
        
        # Store references
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model, tokenizer
    
    def load_teacher_model(self) -> Tuple[torch.nn.Module, object]:
        """
        Load pre-trained 7B teacher model (MinhQuy24/Qwen-7B-Vi-VQA)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        return self.load_model(
            MODELS["7B-trained"].model_name,
            load_in_4bit=False,  # Load full precision for teacher
        )
    
    def load_student_model(self) -> Tuple[torch.nn.Module, object]:
        """
        Load pre-trained 2B student model (MinhQuy24/Qwen-2B-ViVQA)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        return self.load_model(
            MODELS["2B-distilled"].model_name,
            load_in_4bit=True,
        )
    
    def load_base_model(self) -> Tuple[torch.nn.Module, object]:
        """
        Load base 7B model for training (unsloth/Qwen2-VL-7B-Instruct)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = MODELS["7B-base"].model_name
        
        if HAS_UNSLOTH:
            print(f"Loading base model with Unsloth: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            print(f"  Base model loaded with Unsloth!")
            return model, tokenizer
        else:
            print(f"Unsloth not available, using standard transformers loader")
            return self.load_model(model_name, load_in_4bit=True)
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get loaded model by name"""
        return self.models.get(model_name)
    
    def get_tokenizer(self, model_name: str) -> Optional[object]:
        """Get tokenizer by name"""
        return self.tokenizers.get(model_name)
    
    def unload_model(self, model_name: str) -> None:
        """Unload model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            print(f"Model '{model_name}' unloaded")
        
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
    
    def unload_all(self) -> None:
        """Unload all models"""
        self.models.clear()
        self.tokenizers.clear()
        print("All models unloaded")
    
    def list_loaded_models(self) -> list:
        """List all currently loaded models"""
        return list(self.models.keys())
    
    @staticmethod
    def list_available_models() -> dict:
        """List all available pre-configured models"""
        return {
            model_id: {
                "name": config.model_name,
                "params": config.params,
                "purpose": config.purpose,
            }
            for model_id, config in MODELS.items()
        }
    
    @staticmethod
    def print_available_models() -> None:
        """Print available models info"""
        models_info = VQAModelLoader.list_available_models()
        
        print(f"\n{'='*70}")
        print("Available Pre-configured Models")
        print(f"{'='*70}")
        
        for model_id, info in models_info.items():
            print(f"\n📦 {model_id}")
            print(f"   Model: {info['name']}")
            print(f"   Size: {info['params']}")
            print(f"   Purpose: {info['purpose']}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    
    # Print available models
    VQAModelLoader.print_available_models()
    
    # Initialize loader
    loader = VQAModelLoader(device="cuda")
    
    # Load teacher model (7B)
    print("Loading teacher model...")
    teacher_model, teacher_tokenizer = loader.load_teacher_model()
    
    # Load student model (2B)
    print("\nLoading student model...")
    student_model, student_tokenizer = loader.load_student_model()
    
    # List loaded models
    print(f"\nLoaded models: {loader.list_loaded_models()}")

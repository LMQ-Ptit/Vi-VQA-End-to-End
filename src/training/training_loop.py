"""
Main training loop for Vi-VQA
Uses SFTTrainer from TRL library
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import TrainingArguments
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from config_utils import ConfigLoader
from trainer_utils import VQATrainerCallback, TrainingHelper, CheckpointManager


class VQATrainer:
    """Trainer for Vision-Language VQA models"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset: Optional = None,
        config_dir: str = "./configs",
        output_dir: str = "./outputs",
    ):
        """
        Initialize VQA trainer
        
        Args:
            model: Language model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config_dir: Configuration directory
            output_dir: Output directory for checkpoints
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Load configurations
        self.config_loader = ConfigLoader(config_dir)
        self.training_config = self.config_loader.get_training_config()
        
        # Setup environment
        self.config_loader.setup_environment(self.training_config)
        
        # Trainer will be initialized in setup_trainer()
        self.trainer = None
        self.training_args = None
    
    def setup_trainer(self) -> None:
        """Setup SFTTrainer with configurations"""
        
        print(f"\n{'='*60}")
        print("Setting up SFTTrainer")
        print(f"{'='*60}\n")
        
        # Create training arguments
        training_args_kwargs = ConfigLoader.create_training_args(self.training_config)
        training_args_kwargs["output_dir"] = self.output_dir
        
        self.training_args = TrainingArguments(**training_args_kwargs)
        
        # Print config
        TrainingHelper.print_training_config(training_args_kwargs)
        
        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.training_args,
            packing=False,  # Don't pack sequences for vision-language models
            max_seq_length=2048,
            callbacks=[VQATrainerCallback(self.output_dir)],
        )
        
        # Setup data collator for vision-language models
        self.trainer.data_collator = DataCollatorForVisionLanguageModeling(
            processor=self.tokenizer,
            max_length=1024,
        )
        
        print("SFTTrainer ready!")
    
    def train(self, resume_from_checkpoint: bool = True) -> Dict[str, float]:
        """
        Start training
        
        Args:
            resume_from_checkpoint: Whether to resume from checkpoint if available
            
        Returns:
            Training results
        """
        
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
        
        # Try to resume from checkpoint
        if resume_from_checkpoint:
            resumed = CheckpointManager.resume_from_checkpoint(
                self.trainer,
                self.output_dir
            )
            if resumed:
                return self.trainer.state.log_history[-1]
        
        # Start fresh training
        print("▶️  Starting fresh training...")
        train_result = self.trainer.train()
        
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}\n")
        
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on eval_dataset
        
        Returns:
            Evaluation metrics
        """
        
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        
        if self.eval_dataset is None:
            print("No evaluation dataset provided")
            return {}
        
        print(f"\n{'='*60}")
        print("Running Evaluation")
        print(f"{'='*60}\n")
        
        eval_results = self.trainer.evaluate()
        
        print(f"\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
        
        return eval_results
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save trained model
        
        Args:
            output_dir: Directory to save model (uses self.output_dir by default)
        """
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "final_lora")
        
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized")
        
        print(f"\n{'='*60}")
        print(f"Saving model to {output_dir}")
        print(f"{'='*60}\n")
        
        self.trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
    
    def save_tokenizer(self, output_dir: Optional[str] = None) -> None:
        """
        Save tokenizer
        
        Args:
            output_dir: Directory to save tokenizer
        """
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "final_lora")
        
        print(f"Saving tokenizer to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")
    
    def get_trainer(self):
        """Get the underlying trainer object"""
        return self.trainer
    
    def print_training_summary(self) -> None:
        """Print training summary"""
        
        if self.trainer is None:
            print("Trainer not initialized")
            return
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Global Step: {self.trainer.state.global_step}")
        print(f"Epoch: {self.trainer.state.epoch:.1f}")
        print(f"Total Training Loss: {self.trainer.state.best_loss:.4f}")
        print(f"{'='*60}\n")


class DistillationTrainer:
    """Trainer for Knowledge Distillation (Teacher -> Student)"""
    
    def __init__(
        self,
        student_model,
        student_tokenizer,
        teacher_model,
        teacher_tokenizer,
        train_dataset,
        eval_dataset: Optional = None,
        config_dir: str = "./configs",
        output_dir: str = "./outputs-2B-distilled",
    ):
        """
        Initialize distillation trainer
        
        Args:
            student_model: Student model (2B) to train
            student_tokenizer: Student tokenizer
            teacher_model: Teacher model (7B) for knowledge
            teacher_tokenizer: Teacher tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config_dir: Configuration directory
            output_dir: Output directory for checkpoints
        """
        
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Load distillation config
        self.config_loader = ConfigLoader(config_dir)
        self.distillation_config = self._load_distillation_config()
        self.training_config = self.config_loader.get_training_config()
        
        # Setup environment
        self.config_loader.setup_environment(self.training_config)
        
        self.trainer = None
    
    def _load_distillation_config(self) -> Dict[str, Any]:
        """Load distillation configuration"""
        try:
            return self.config_loader.load_yaml("distillation_config.yaml")
        except FileNotFoundError:
            print("distillation_config.yaml not found, using defaults")
            return {
                "distillation": {
                    "temperature": 4.0,
                    "alpha": 0.5,
                    "training": {}
                }
            }
    
    def setup_trainer(self) -> None:
        """Setup trainer for distillation"""
        
        print(f"\n{'='*60}")
        print("Setting up Distillation Trainer")
        print(f"{'='*60}\n")
        
        # Get distillation hyperparams
        distillation_config = self.distillation_config.get("distillation", {})
        self.temperature = distillation_config.get("temperature", 4.0)
        self.alpha = distillation_config.get("alpha", 0.5)
        
        print(f"Temperature: {self.temperature}")
        print(f"Alpha (KD loss weight): {self.alpha}")
        print()
        
        # Create training arguments
        training_config = distillation_config.get("training", {})
        training_args_kwargs = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
            "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 1),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 8),
            "warmup_steps": training_config.get("warmup_steps", 10),
            "num_train_epochs": training_config.get("num_train_epochs", 2),
            "learning_rate": training_config.get("learning_rate", 5e-5),
            "bf16": training_config.get("bf16", True),
            "fp16": training_config.get("fp16", False),
            "optim": training_config.get("optim", "adamw_8bit"),
            "logging_steps": training_config.get("logging_steps", 1),
            "eval_steps": training_config.get("eval_steps", 50),
            "save_steps": training_config.get("save_steps", 200),
            "save_strategy": training_config.get("save_strategy", "steps"),
            "remove_unused_columns": training_config.get("remove_unused_columns", False),
        }
        
        training_args = TrainingArguments(**training_args_kwargs)
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.student_model,
            tokenizer=self.student_tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            packing=False,
            max_seq_length=2048,
            callbacks=[VQATrainerCallback(self.output_dir)],
        )
        
        # Setup data collator
        self.trainer.data_collator = DataCollatorForVisionLanguageModeling(
            processor=self.student_tokenizer,
            max_length=1024,
        )
        
        print("Distillation Trainer ready!")
    
    def train(self) -> Dict[str, float]:
        """Start distillation training"""
        
        if self.trainer is None:
            self.setup_trainer()
        
        print(f"\n{'='*60}")
        print("Starting Distillation Training")
        print(f"{'='*60}\n")
        print(f"Student (2B) learning from Teacher (7B)...\n")
        
        train_result = self.trainer.train()
        
        print(f"\n{'='*60}")
        print("Distillation Training Completed!")
        print(f"{'='*60}\n")
        
        return train_result
    
    def save_student_model(self, output_dir: Optional[str] = None) -> None:
        """Save distilled student model"""
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "final_lora")
        
        print(f"Saving student model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.student_tokenizer.save_pretrained(output_dir)
        print(f"Student model saved to {output_dir}")


if __name__ == "__main__":
    print("Training loop utilities ready to use")

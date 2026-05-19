"""
Training utilities for Vi-VQA
Includes callbacks, metrics, and helper functions
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import TrainerCallback, TrainerState, TrainerControl
import json
from pathlib import Path


class VQATrainerCallback(TrainerCallback):
    """Custom callback for VQA training"""
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize callback
        
        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = []
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at end of each training step"""
        
        if state.global_step % args.logging_steps == 0:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
            }
            
            if "eval_loss" in state.log_history[-1]:
                log_entry["eval_loss"] = state.log_history[-1]["eval_loss"]
            
            self.training_history.append(log_entry)
    
    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called when checkpoint is saved"""
        print(f"Checkpoint saved at step {state.global_step}")
    
    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at end of training"""
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        print(f"Total steps: {state.global_step}")
        print(f"Total epochs: {state.epoch}")


class MetricsCalculator:
    """Calculate evaluation metrics for VQA"""
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Exact match score (0-1)
        """
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        matches = 0
        for pred, ref in zip(predictions, references):
            if str(pred).strip().lower() == str(ref).strip().lower():
                matches += 1
        
        return matches / len(predictions)
    
    @staticmethod
    def rouge_l(predictions: List[str], references: List[str]) -> float:
        """
        Calculate ROUGE-L score (F1)
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(
                    str(ref).strip().lower(),
                    str(pred).strip().lower()
                )
                scores.append(score['rougeL'].fmeasure)
            
            return np.mean(scores) if scores else 0.0
        
        except ImportError:
            print("Warning: rouge_score not installed, using BERTScore instead")
            return MetricsCalculator.bertscore_f1(predictions, references)
    
    @staticmethod
    def bertscore_f1(predictions: List[str], references: List[str]) -> float:
        """
        Calculate BERTScore F1
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Average BERTScore F1 (0-1)
        """
        
        try:
            from bert_score import score
            
            P, R, F1 = score(
                predictions,
                references,
                lang='vi',
                verbose=False
            )
            
            return F1.mean().item()
        
        except ImportError:
            print("Warning: bert_score not installed")
            return 0.0
    
    @staticmethod
    def calculate_all_metrics(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary with metric scores
        """
        
        return {
            "exact_match": MetricsCalculator.exact_match(predictions, references),
            "rouge_l": MetricsCalculator.rouge_l(predictions, references),
            "bertscore_f1": MetricsCalculator.bertscore_f1(predictions, references),
        }


class TrainingHelper:
    """Helper utilities for training"""
    
    @staticmethod
    def print_training_config(args: Dict[str, Any]) -> None:
        """Print training configuration"""
        
        print(f"\n{'='*60}")
        print("Training Configuration")
        print(f"{'='*60}")
        
        for key, value in args.items():
            print(f"{key:30s}: {value}")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def get_device() -> str:
        """Get available device (cuda or cpu)"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """Get GPU memory usage stats"""
        
        if not torch.cuda.is_available():
            return {"message": "CUDA not available"}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "cached_gb": torch.cuda.memory_cached() / 1e9,
        }
    
    @staticmethod
    def print_gpu_info() -> None:
        """Print GPU information"""
        
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
        
        print(f"\n{'='*60}")
        print("GPU Information")
        print(f"{'='*60}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        
        print(f"\nCurrent Memory Usage:")
        memory = TrainingHelper.get_gpu_memory_usage()
        for key, value in memory.items():
            print(f"  {key}: {value:.2f} GB")
        
        print(f"{'='*60}\n")


class CheckpointManager:
    """Manage training checkpoints"""
    
    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> Optional[str]:
        """
        Find latest checkpoint in output directory
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Path to latest checkpoint or None
        """
        
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return None
        
        checkpoint_dirs = [
            d for d in output_path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        
        if not checkpoint_dirs:
            return None
        
        # Get latest checkpoint by step number
        latest = max(
            checkpoint_dirs,
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        return str(latest)
    
    @staticmethod
    def resume_from_checkpoint(trainer, output_dir: str) -> bool:
        """
        Resume training from latest checkpoint if available
        
        Args:
            trainer: Trainer object
            output_dir: Output directory
            
        Returns:
            True if resumed, False otherwise
        """
        
        latest_checkpoint = CheckpointManager.find_latest_checkpoint(output_dir)
        
        if latest_checkpoint:
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            print(f"▶️  Resuming training from checkpoint...")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
            return True
        
        return False


if __name__ == "__main__":
    # Example usage
    print("Training utilities ready to use")
    
    # Print GPU info
    TrainingHelper.print_gpu_info()
    
    # Calculate metrics example
    predictions = ["Một chiếc xe", "Hai người", "Ba ngôi nhà"]
    references = ["Một chiếc ô tô", "Hai cô gái", "Ba tòa nhà"]
    
    metrics = MetricsCalculator.calculate_all_metrics(predictions, references)
    print("\nMetrics Example:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

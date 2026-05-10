"""
Script train teacher model (Qwen2-VL-7B)
Chạy: python scripts/train_teacher.py
"""
import os
import sys

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.teacher.model import load_teacher_model, load_teacher_processor, get_answer
from src.teacher.dataset import load_vqa_split, format_vqa_example
from src.teacher.trainer import load_model_and_tokenizer, create_vqa_trainer, train_with_checkpoint, save_final_model
from datasets import Dataset
import yaml

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "teacher.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("="*50)
    print("Training Teacher Model (Qwen2-VL-7B)")
    print("="*50)

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"]
    )
    print("Model loaded successfully!")

    # Load data
    print("\n[2/5] Loading dataset...")
    train_data = load_vqa_split(
        config["data"]["train_path"],
        config["data"]["train_images"]
    )
    dev_data = load_vqa_split(
        config["data"]["dev_path"],
        config["data"]["dev_images"]
    )

    # Format dataset
    print("\n[3/5] Formatting dataset...")
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_vqa_example, batched=True)

    dev_dataset = Dataset.from_list(dev_data)
    dev_dataset = dev_dataset.map(format_vqa_example, batched=True)
    print(f"Train: {len(train_dataset)} samples, Dev: {len(dev_dataset)} samples")

    # Create trainer
    print("\n[4/5] Creating trainer...")
    trainer = create_vqa_trainer(
        model,
        train_dataset,
        dev_dataset,
        tokenizer
    )

    # Train
    print("\n[5/5] Starting training...")
    trainer.train()

    # Save model
    save_dir = "models/teacher"
    os.makedirs(save_dir, exist_ok=True)
    save_final_model(model, tokenizer, save_dir)

    # Evaluate
    print("\nEvaluating on dev set...")
    trainer.evaluate(eval_dataset=dev_dataset)

    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {save_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
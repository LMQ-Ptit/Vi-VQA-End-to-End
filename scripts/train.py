#!/usr/bin/env python3
"""
Training script for Vi-VQA base model (7B).

Trains Qwen2-VL-7B-Instruct on Vietnamese VQA dataset using LoRA fine-tuning.

Usage:
    python train.py --config-dir ../configs --output-dir ../outputs-7B
    python train.py --config-dir ../configs --output-dir ../outputs-7B --resume
    python train.py --config-dir ../configs --output-dir ../outputs-7B --push-to-hub
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VQAModelLoader
from src.data import VQADataLoader, prepare_vqa_dataset
from src.training import VQATrainer, TrainingHelper, CheckpointManager, ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train Vi-VQA base model (7B) on Vietnamese VQA dataset'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='./configs',
        help='Path to configs directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs-7B',
        help='Output directory for checkpoints and final model'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint if available'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Maximum number of training samples (None = all)'
    )
    parser.add_argument(
        '--max-eval-samples',
        type=int,
        default=None,
        help='Maximum number of eval samples (None = all)'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push trained model to HuggingFace Hub'
    )
    parser.add_argument(
        '--hub-model-id',
        type=str,
        default='MinhQuy24/Qwen-7B-Vi-VQA',
        help='HuggingFace Hub model ID for pushing'
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Vi-VQA Base Model Training (7B)")
    logger.info("=" * 80)
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Resume from checkpoint: {args.resume}")
    if args.push_to_hub:
        logger.info(f"Push to HuggingFace Hub: {args.hub_model_id}")

    # ===== STEP 1: Print GPU Info =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: GPU Information")
    logger.info("=" * 80)
    TrainingHelper.print_gpu_info()

    # ===== STEP 2: Load Model =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading Base Model")
    logger.info("=" * 80)
    logger.info("Loading unsloth/Qwen2-VL-7B-Instruct...")
    loader = VQAModelLoader()
    model, tokenizer = loader.load_base_model()
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Model dtype: {model.dtype}")

    # ===== STEP 3: Load Dataset =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading Dataset")
    logger.info("=" * 80)
    logger.info("Loading Vietnamese VQA dataset (MinhQuy24/vlsp2023-vqa-dataset)...")
    data_loader = VQADataLoader()
    
    # Load splits
    train_dataset = data_loader.get_train()
    dev_dataset = data_loader.get_dev()
    
    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Dev set size: {len(dev_dataset)}")
    
    # Print sample
    logger.info("\nSample from dataset:")
    data_loader.print_sample(train_dataset, idx=0)

    # ===== STEP 4: Prepare Dataset =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Preparing Dataset")
    logger.info("=" * 80)
    
    max_train = args.max_train_samples or len(train_dataset)
    max_eval = args.max_eval_samples or len(dev_dataset)
    
    logger.info(f"Preparing training set (max {max_train} samples)...")
    train_prepared = prepare_vqa_dataset(
        train_dataset, 
        tokenizer, 
        max_samples=max_train
    )
    logger.info(f"Train set prepared: {len(train_prepared)} samples")
    
    logger.info(f"Preparing eval set (max {max_eval} samples)...")
    dev_prepared = prepare_vqa_dataset(
        dev_dataset,
        tokenizer,
        max_samples=max_eval
    )
    logger.info(f"Dev set prepared: {len(dev_prepared)} samples")

    # ===== STEP 5: Create Trainer =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Setting up Trainer")
    logger.info("=" * 80)
    
    trainer = VQATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_prepared,
        eval_dataset=dev_prepared,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    
    trainer.setup_trainer()
    logger.info("Trainer setup complete")

    # ===== STEP 6: Train Model =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Training")
    logger.info("=" * 80)
    
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = CheckpointManager.find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"\nTraining complete!")
    logger.info(f"  Final training loss: {train_result.training_loss:.4f}")

    # ===== STEP 7: Evaluate =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Evaluation")
    logger.info("=" * 80)
    
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation complete!")
    logger.info(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    # ===== STEP 8: Save Model =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Saving Model")
    logger.info("=" * 80)
    
    final_model_path = f"{args.output_dir}/final_lora"
    logger.info(f"Saving model to {final_model_path}...")
    trainer.save_model(final_model_path)
    trainer.save_tokenizer(final_model_path)
    logger.info(f"Model saved!")

    # ===== STEP 9: Print Summary =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Training Summary")
    logger.info("=" * 80)
    trainer.print_training_summary()

    # ===== STEP 10: Push to Hub (Optional) =====
    if args.push_to_hub:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 10: Pushing to HuggingFace Hub")
        logger.info("=" * 80)
        logger.info(f"Pushing model to {args.hub_model_id}...")
        try:
            model.push_to_hub(args.hub_model_id, token=True)
            tokenizer.push_to_hub(args.hub_model_id, token=True)
            logger.info(f"Model pushed to {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("You can manually push later with:")
            logger.info(f"  huggingface-cli upload {args.hub_model_id} {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

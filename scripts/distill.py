#!/usr/bin/env python3
"""
Knowledge Distillation script for Vi-VQA (7B -> 2B).

Trains Qwen2-VL-2B-Instruct as student using Qwen-7B-Vi-VQA as teacher.

Usage:
    python distill.py --config-dir ../configs --output-dir ../outputs-2B-distilled
    python distill.py --config-dir ../configs --output-dir ../outputs-2B-distilled --resume
    python distill.py --config-dir ../configs --output-dir ../outputs-2B-distilled --push-to-hub
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VQAModelLoader
from src.data import VQADataLoader, prepare_vqa_dataset
from src.training import DistillationTrainer, TrainingHelper, CheckpointManager, ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation: Train 2B student from 7B teacher'
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
        default='./outputs-2B-distilled',
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
        help='Push distilled student model to HuggingFace Hub'
    )
    parser.add_argument(
        '--hub-model-id',
        type=str,
        default='MinhQuy24/Qwen-2B-ViVQA',
        help='HuggingFace Hub model ID for pushing'
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Vi-VQA Knowledge Distillation (7B Teacher -> 2B Student)")
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

    # ===== STEP 2: Load Models =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading Teacher & Student Models")
    logger.info("=" * 80)
    
    loader = VQAModelLoader()
    
    logger.info("Loading teacher model (MinhQuy24/Qwen-7B-Vi-VQA)...")
    teacher_model, teacher_tokenizer = loader.load_teacher_model()
    logger.info(f"Teacher loaded: {teacher_model.__class__.__name__}")
    
    logger.info("Loading student model (Qwen/Qwen2-VL-2B-Instruct)...")
    student_model, student_tokenizer = loader.load_student_model()
    logger.info(f"Student loaded: {student_model.__class__.__name__}")
    
    logger.info(f"  Teacher model size: {sum(p.numel() for p in teacher_model.parameters()) / 1e9:.2f}B params")
    logger.info(f"  Student model size: {sum(p.numel() for p in student_model.parameters()) / 1e9:.2f}B params")

    # ===== STEP 3: Load Dataset =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading Dataset")
    logger.info("=" * 80)
    logger.info("Loading Vietnamese VQA dataset (MinhQuy24/vlsp2023-vqa-dataset)...")
    
    data_loader = VQADataLoader()
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
        student_tokenizer,
        max_samples=max_train
    )
    logger.info(f"Train set prepared: {len(train_prepared)} samples")
    
    logger.info(f"Preparing eval set (max {max_eval} samples)...")
    dev_prepared = prepare_vqa_dataset(
        dev_dataset,
        student_tokenizer,
        max_samples=max_eval
    )
    logger.info(f"Dev set prepared: {len(dev_prepared)} samples")

    # ===== STEP 5: Create Distillation Trainer =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Setting up Distillation Trainer")
    logger.info("=" * 80)
    
    # Load distillation config to show KD parameters
    config_loader = ConfigLoader(args.config_dir)
    distillation_config = config_loader.get_distillation_config()
    
    logger.info(f"Knowledge Distillation Parameters:")
    logger.info(f"  Temperature: {distillation_config['distillation']['temperature']}")
    logger.info(f"  Alpha (KD loss weight): {distillation_config['distillation']['alpha']}")
    logger.info(f"  Method: Response-based (student learns to match teacher outputs)")
    
    distill_trainer = DistillationTrainer(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        train_dataset=train_prepared,
        eval_dataset=dev_prepared,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    
    distill_trainer.setup_trainer()
    logger.info("Distillation trainer setup complete")

    # ===== STEP 6: Train with Distillation =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Knowledge Distillation Training")
    logger.info("=" * 80)
    
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = CheckpointManager.find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    train_result = distill_trainer.train()
    logger.info(f"\nDistillation training complete!")
    logger.info(f"  Final training loss: {train_result.training_loss:.4f}")

    # ===== STEP 7: Evaluate Student =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Evaluation")
    logger.info("=" * 80)
    
    eval_result = distill_trainer.evaluate()
    logger.info(f"Evaluation complete!")
    logger.info(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    # ===== STEP 8: Save Student Model =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Saving Student Model")
    logger.info("=" * 80)
    
    final_model_path = f"{args.output_dir}/final_lora"
    logger.info(f"Saving distilled student model to {final_model_path}...")
    distill_trainer.save_student_model(final_model_path)
    logger.info(f"Student model saved!")

    # ===== STEP 9: Push to Hub (Optional) =====
    if args.push_to_hub:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Pushing to HuggingFace Hub")
        logger.info("=" * 80)
        logger.info(f"Pushing student model to {args.hub_model_id}...")
        try:
            student_model.push_to_hub(args.hub_model_id, token=True)
            student_tokenizer.push_to_hub(args.hub_model_id, token=True)
            logger.info(f"Student model pushed to {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("You can manually push later with:")
            logger.info(f"  huggingface-cli upload {args.hub_model_id} {final_model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Knowledge Distillation pipeline complete!")
    logger.info(f"2B student model is ready: {final_model_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Inference script for Vi-VQA models.

Run inference with either:
- Teacher model (7B): MinhQuy24/Qwen-7B-Vi-VQA
- Student model (2B): MinhQuy24/Qwen-2B-ViVQA

Supports:
- Single image + question inference
- Batch inference from JSON file
- Comparison between teacher and student

Usage:
    # Single inference
    python inference.py --model-type teacher --image-path image.jpg --question "Đây là gì?"
    
    # Batch inference from JSON
    python inference.py --model-type teacher --input-json data.json --output-json predictions.json
    
    # Compare teacher vs student
    python inference.py --model-type both --input-json data.json --output-json comparison.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VQAModelLoader
from src.models.inference import VQAInference, ComparativeInference
from src.models.model_wrapper import TeacherModel, StudentModel, ModelPair

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def infer_single(model_type, image_path, question):
    """Run single image inference."""
    logger.info("=" * 80)
    logger.info("Single Image Inference")
    logger.info("=" * 80)
    
    loader = VQAModelLoader()
    
    if model_type == 'teacher':
        logger.info("Loading teacher model (7B)...")
        model, tokenizer = loader.load_teacher_model()
        teacher_wrapper = TeacherModel(model, tokenizer)
        inference = VQAInference(teacher_wrapper)
        
    elif model_type == 'student':
        logger.info("Loading student model (2B)...")
        model, tokenizer = loader.load_student_model()
        student_wrapper = StudentModel(model, tokenizer)
        inference = VQAInference(student_wrapper)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run inference
    logger.info(f"Image: {image_path}")
    logger.info(f"Question: {question}")
    logger.info("Running inference...")
    
    result = inference.predict_single(image_path, question)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Result:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
    
    return result


def infer_batch(model_type, input_json, output_json=None):
    """Run batch inference from JSON file."""
    logger.info("=" * 80)
    logger.info("Batch Inference")
    logger.info("=" * 80)
    
    # Load input JSON
    logger.info(f"Loading input data from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Total samples: {len(data)}")
    
    # Load model
    loader = VQAModelLoader()
    
    if model_type == 'teacher':
        logger.info("Loading teacher model (7B)...")
        model, tokenizer = loader.load_teacher_model()
        teacher_wrapper = TeacherModel(model, tokenizer)
        inference = VQAInference(teacher_wrapper)
        
    elif model_type == 'student':
        logger.info("Loading student model (2B)...")
        model, tokenizer = loader.load_student_model()
        student_wrapper = StudentModel(model, tokenizer)
        inference = VQAInference(student_wrapper)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run batch inference
    logger.info("Running batch inference...")
    predictions = inference.predict_from_json(input_json)
    
    logger.info(f"Batch inference complete! Processed {len(predictions)} samples")
    
    # Save predictions
    if output_json:
        logger.info(f"Saving predictions to {output_json}...")
        inference.save_predictions(predictions, output_json)
        logger.info(f"Saved!")
    
    # Print first 5 results
    logger.info(f"\nFirst 5 predictions:")
    for i, pred in enumerate(predictions[:5]):
        logger.info(f"  {i+1}. {pred['answer']}")
    
    return predictions


def compare_models(input_json, output_json=None):
    """Compare teacher vs student models."""
    logger.info("=" * 80)
    logger.info("Teacher vs Student Comparison")
    logger.info("=" * 80)
    
    # Load input JSON
    logger.info(f"Loading input data from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Total samples: {len(data)}")
    
    # Load models
    loader = VQAModelLoader()
    
    logger.info("Loading teacher model (7B)...")
    teacher_model, teacher_tokenizer = loader.load_teacher_model()
    teacher_wrapper = TeacherModel(teacher_model, teacher_tokenizer)
    
    logger.info("Loading student model (2B)...")
    student_model, student_tokenizer = loader.load_student_model()
    student_wrapper = StudentModel(student_model, student_tokenizer)
    
    # Create model pair
    model_pair = ModelPair(teacher_wrapper, student_wrapper)
    
    # Comparative inference
    comparative = ComparativeInference(model_pair)
    
    logger.info("Running comparative inference...")
    comparisons = comparative.compare_from_json(input_json)
    
    logger.info(f"Comparison complete! Processed {len(comparisons)} samples")
    
    # Save comparisons
    if output_json:
        logger.info(f"Saving comparisons to {output_json}...")
        comparative.save_comparison(comparisons, output_json)
        logger.info(f"Saved!")
    
    # Print first 3 comparisons
    logger.info(f"\nFirst 3 comparisons (Teacher vs Student):")
    for i, comp in enumerate(comparisons[:3]):
        teacher_ans = comp.get('teacher_answer', 'N/A')
        student_ans = comp.get('student_answer', 'N/A')
        match = 'yes' if teacher_ans.lower() == student_ans.lower() else 'no'
        logger.info(f"  {i+1}. [Teacher] {teacher_ans} vs [Student] {student_ans} {match}")
    
    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with Vi-VQA models'
    )
    
    # Model selection
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['teacher', 'student', 'both'],
        default='teacher',
        help='Model type: teacher (7B), student (2B), or both (compare)'
    )
    
    # Single inference
    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to image for single inference'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question for single inference'
    )
    
    # Batch inference
    parser.add_argument(
        '--input-json',
        type=str,
        help='Input JSON file with image paths and questions'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output JSON file for predictions/comparisons'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image_path and args.question:
        # Single inference mode
        if args.input_json:
            logger.error("Cannot use both --image-path/--question and --input-json")
            return
        
        result = infer_single(args.model_type, args.image_path, args.question)
        return
    
    elif args.input_json:
        # Batch inference mode
        if args.model_type == 'both':
            # Comparison mode
            comparisons = compare_models(args.input_json, args.output_json)
        else:
            # Single model batch inference
            predictions = infer_batch(args.model_type, args.input_json, args.output_json)
        return
    
    else:
        logger.error("Must provide either:")
        logger.error("  1. --image-path and --question (for single inference)")
        logger.error("  2. --input-json (for batch inference)")
        parser.print_help()
        return


if __name__ == '__main__':
    main()

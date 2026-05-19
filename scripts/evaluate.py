#!/usr/bin/env python3
"""
Evaluation script for Vi-VQA models.

Evaluates trained models on the test set and calculates metrics.

Metrics:
- Exact Match: Percentage of exact string matches
- ROUGE-L: F1 score based on longest common subsequence
- BERTScore F1: Contextual semantic similarity using BERT embeddings

Usage:
    # Evaluate teacher model on test set
    python evaluate.py --model-type teacher --output-file eval_results.json
    
    # Evaluate student model
    python evaluate.py --model-type student --output-file eval_results.json
    
    # Compare teacher vs student
    python evaluate.py --model-type both --output-file comparison.json
    
    # Custom test dataset
    python evaluate.py --model-type teacher --test-json custom_test.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VQAModelLoader
from src.data import VQADataLoader
from src.models.inference import VQAInference, ComparativeInference
from src.models.model_wrapper import TeacherModel, StudentModel, ModelPair
from src.training import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_dataset_to_json(dataset, output_path):
    """Convert dataset to JSON format for inference."""
    logger.info(f"Converting dataset to JSON format...")
    
    data = []
    for sample in dataset:
        item = {
            'image': sample.get('image'),  # PIL Image
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'id': sample.get('id', len(data))
        }
        data.append(item)
    
    # Save as JSON (convert PIL images to paths/metadata)
    json_data = []
    for item in data:
        json_data.append({
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
        })
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(json_data)} samples to {output_path}")
    return data


def evaluate_model(model_type, test_dataset=None, test_json=None):
    """Evaluate single model."""
    logger.info("=" * 80)
    logger.info(f"Evaluating {model_type.upper()} Model")
    logger.info("=" * 80)
    
    # Load model
    loader = VQAModelLoader()
    
    if model_type == 'teacher':
        logger.info("Loading teacher model (7B)...")
        model, tokenizer = loader.load_teacher_model()
        wrapper = TeacherModel(model, tokenizer)
        
    elif model_type == 'student':
        logger.info("Loading student model (2B)...")
        model, tokenizer = loader.load_student_model()
        wrapper = StudentModel(model, tokenizer)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create inference
    inference = VQAInference(wrapper)
    
    # Load test dataset
    if test_json:
        logger.info(f"Loading test data from {test_json}...")
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        num_samples = len(test_data)
    else:
        logger.info("Loading test dataset from HuggingFace...")
        data_loader = VQADataLoader()
        test_dataset = data_loader.get_test()
        num_samples = len(test_dataset)
    
    logger.info(f"Test set size: {num_samples}")
    
    # Run inference
    logger.info("Running inference on test set...")
    if test_json:
        predictions = inference.predict_from_json(test_json)
        # Extract ground truth
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        references = [item.get('answer', '') for item in test_data]
    else:
        predictions = []
        references = []
        for sample in test_dataset:
            question = sample.get('question', '')
            image = sample.get('image')
            answer = sample.get('answer', '')
            
            pred = inference.predict_single(image, question)
            predictions.append(pred['answer'])
            references.append(answer)
    
    logger.info(f"✓ Inference complete! Processed {len(predictions)} samples")
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    metrics = MetricsCalculator.calculate_all_metrics(predictions, references)
    
    # Print results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Model: {model_type.upper()}")
    logger.info(f"Samples: {len(predictions)}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
    logger.info(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
    
    results = {
        'model': model_type,
        'num_samples': len(predictions),
        'metrics': metrics,
        'predictions': predictions,
        'references': references,
    }
    
    return results


def compare_models_eval(test_dataset=None, test_json=None):
    """Compare teacher vs student on test set."""
    logger.info("=" * 80)
    logger.info("Comparing Teacher vs Student Models")
    logger.info("=" * 80)
    
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
    
    # Create comparative inference
    comparative = ComparativeInference(model_pair)
    
    # Load test dataset
    if test_json:
        logger.info(f"Loading test data from {test_json}...")
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        num_samples = len(test_data)
    else:
        logger.info("Loading test dataset from HuggingFace...")
        data_loader = VQADataLoader()
        test_dataset = data_loader.get_test()
        num_samples = len(test_dataset)
    
    logger.info(f"Test set size: {num_samples}")
    
    # Run comparison
    logger.info("Running comparative inference...")
    if test_json:
        comparisons = comparative.compare_from_json(test_json)
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        references = [item.get('answer', '') for item in test_data]
    else:
        comparisons = []
        references = []
        for sample in test_dataset:
            question = sample.get('question', '')
            image = sample.get('image')
            answer = sample.get('answer', '')
            
            comp = comparative.compare_single(image, question)
            comparisons.append(comp)
            references.append(answer)
    
    logger.info(f"✓ Comparison complete! Processed {len(comparisons)} samples")
    
    # Extract predictions
    teacher_preds = [c['teacher_answer'] for c in comparisons]
    student_preds = [c['student_answer'] for c in comparisons]
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    teacher_metrics = MetricsCalculator.calculate_all_metrics(teacher_preds, references)
    student_metrics = MetricsCalculator.calculate_all_metrics(student_preds, references)
    
    # Count matches between teacher and student
    matches = sum(1 for t, s in zip(teacher_preds, student_preds) 
                  if t.lower() == s.lower())
    
    # Print results
    logger.info("=" * 80)
    logger.info("Comparison Results")
    logger.info("=" * 80)
    logger.info(f"Samples: {len(comparisons)}")
    logger.info(f"\nTeacher (7B) Metrics:")
    logger.info(f"  Exact Match: {teacher_metrics['exact_match']:.4f}")
    logger.info(f"  ROUGE-L: {teacher_metrics['rouge_l']:.4f}")
    logger.info(f"  BERTScore F1: {teacher_metrics['bertscore_f1']:.4f}")
    
    logger.info(f"\nStudent (2B) Metrics:")
    logger.info(f"  Exact Match: {student_metrics['exact_match']:.4f}")
    logger.info(f"  ROUGE-L: {student_metrics['rouge_l']:.4f}")
    logger.info(f"  BERTScore F1: {student_metrics['bertscore_f1']:.4f}")
    
    logger.info(f"\nTeacher-Student Agreement: {matches}/{len(comparisons)} ({100*matches/len(comparisons):.1f}%)")
    
    results = {
        'num_samples': len(comparisons),
        'teacher_metrics': teacher_metrics,
        'student_metrics': student_metrics,
        'agreement': matches / len(comparisons),
        'comparisons': comparisons,
        'references': references,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Vi-VQA models on test set'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['teacher', 'student', 'both'],
        default='teacher',
        help='Model type: teacher (7B), student (2B), or both (compare)'
    )
    
    parser.add_argument(
        '--test-json',
        type=str,
        help='Custom test JSON file (if not provided, loads from HuggingFace)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file to save results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    if args.model_type == 'both':
        results = compare_models_eval(test_json=args.test_json)
    else:
        results = evaluate_model(args.model_type, test_json=args.test_json)
    
    # Save results
    if args.output_file:
        logger.info(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved!")


if __name__ == '__main__':
    main()

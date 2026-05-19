"""
Inference utilities for Vi-VQA models
"""

import torch
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image


class VQAInference:
    """Inference engine for VQA tasks"""
    
    def __init__(self, model_wrapper):
        """
        Initialize inference engine
        
        Args:
            model_wrapper: VisionLanguageModelWrapper instance
        """
        self.model = model_wrapper
    
    def predict_single(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 48,
    ) -> str:
        """
        Predict answer for single image-question pair
        
        Args:
            image: PIL Image
            question: Question text
            max_new_tokens: Max generation length
            
        Returns:
            Predicted answer
        """
        return self.model.answer_question(
            image,
            question,
            max_new_tokens,
        )
    
    def predict_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_new_tokens: int = 48,
    ) -> List[str]:
        """
        Predict answers for batch of image-question pairs
        
        Args:
            images: List of PIL Images
            questions: List of questions
            max_new_tokens: Max generation length
            
        Returns:
            List of predicted answers
        """
        return self.model.generate(
            images=images,
            questions=questions,
            max_new_tokens=max_new_tokens,
        )
    
    def predict_from_file(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 48,
    ) -> str:
        """
        Predict from image file path
        
        Args:
            image_path: Path to image file
            question: Question text
            max_new_tokens: Max generation length
            
        Returns:
            Predicted answer
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict_single(image, question, max_new_tokens)
    
    def predict_from_json(
        self,
        json_path: str,
        max_new_tokens: int = 48,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Predict from JSON file with image paths and questions
        
        Expected JSON format:
        {
            "annotations": [
                {"image_id": 1, "image_path": "...", "question": "..."},
                ...
            ]
        }
        
        Args:
            json_path: Path to JSON file
            max_new_tokens: Max generation length
            limit: Limit number of samples (None = all)
            
        Returns:
            List of predictions with answers
        """
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        annotations = data.get("annotations", [])
        if limit:
            annotations = annotations[:limit]
        
        results = []
        for idx, ann in enumerate(annotations, 1):
            print(f"Processing {idx}/{len(annotations)}...")
            
            image_path = ann.get("image_path")
            question = ann.get("question")
            
            if not image_path or not question:
                continue
            
            try:
                answer = self.predict_from_file(
                    image_path,
                    question,
                    max_new_tokens,
                )
                
                results.append({
                    "image_id": ann.get("image_id"),
                    "image_path": image_path,
                    "question": question,
                    "predicted_answer": answer,
                    "reference_answer": ann.get("answer"),  # If available
                })
            except Exception as e:
                print(f"  ❌ Error processing {image_path}: {e}")
                continue
        
        return results
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_path: str,
    ) -> None:
        """
        Save predictions to JSON file
        
        Args:
            predictions: List of prediction dicts
            output_path: Output JSON file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(predictions)} predictions to {output_path}")
    
    @staticmethod
    def evaluate_predictions(
        predictions: List[Dict],
        metric: str = "exact_match",
    ) -> float:
        """
        Evaluate predictions against reference answers
        
        Args:
            predictions: List of prediction dicts with 'predicted_answer' and 'reference_answer'
            metric: Evaluation metric ("exact_match", "rouge_l", or "bertscore_f1")
            
        Returns:
            Evaluation score
        """
        
        if metric == "exact_match":
            correct = 0
            for pred in predictions:
                if pred.get("reference_answer") is None:
                    continue
                
                if str(pred["predicted_answer"]).strip().lower() == \
                   str(pred["reference_answer"]).strip().lower():
                    correct += 1
            
            return correct / len([p for p in predictions if p.get("reference_answer")]) \
                if any(p.get("reference_answer") for p in predictions) else 0.0
        
        elif metric == "partial_match":
            from difflib import SequenceMatcher
            
            scores = []
            for pred in predictions:
                if pred.get("reference_answer") is None:
                    continue
                
                pred_ans = str(pred["predicted_answer"]).strip().lower()
                ref_ans = str(pred["reference_answer"]).strip().lower()
                
                similarity = SequenceMatcher(None, pred_ans, ref_ans).ratio()
                scores.append(similarity)
            
            return sum(scores) / len(scores) if scores else 0.0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")


class ComparativeInference:
    """Compare predictions from teacher and student models"""
    
    def __init__(self, teacher_wrapper, student_wrapper):
        """
        Initialize comparative inference
        
        Args:
            teacher_wrapper: Teacher model wrapper
            student_wrapper: Student model wrapper
        """
        self.teacher_inference = VQAInference(teacher_wrapper)
        self.student_inference = VQAInference(student_wrapper)
    
    def compare_single(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 48,
    ) -> Dict[str, str]:
        """
        Compare teacher and student predictions for single sample
        
        Args:
            image: PIL Image
            question: Question text
            max_new_tokens: Max generation length
            
        Returns:
            Dictionary with teacher and student predictions
        """
        
        teacher_pred = self.teacher_inference.predict_single(
            image,
            question,
            max_new_tokens,
        )
        
        student_pred = self.student_inference.predict_single(
            image,
            question,
            max_new_tokens,
        )
        
        return {
            "question": question,
            "teacher_answer": teacher_pred,
            "student_answer": student_pred,
        }
    
    def compare_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_new_tokens: int = 48,
    ) -> List[Dict[str, str]]:
        """
        Compare teacher and student predictions for batch
        
        Args:
            images: List of PIL Images
            questions: List of questions
            max_new_tokens: Max generation length
            
        Returns:
            List of comparison dicts
        """
        
        teacher_preds = self.teacher_inference.predict_batch(
            images,
            questions,
            max_new_tokens,
        )
        
        student_preds = self.student_inference.predict_batch(
            images,
            questions,
            max_new_tokens,
        )
        
        comparisons = []
        for q, t_pred, s_pred in zip(questions, teacher_preds, student_preds):
            comparisons.append({
                "question": q,
                "teacher_answer": t_pred,
                "student_answer": s_pred,
            })
        
        return comparisons
    
    def save_comparison(
        self,
        comparisons: List[Dict],
        output_path: str,
    ) -> None:
        """
        Save comparison results to JSON
        
        Args:
            comparisons: List of comparison dicts
            output_path: Output JSON path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(comparisons)} comparisons to {output_path}")


if __name__ == "__main__":
    # Example usage (requires models to be loaded)
    print("Inference utilities ready to use")

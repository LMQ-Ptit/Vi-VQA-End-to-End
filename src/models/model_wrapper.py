"""
Model wrapper classes for Vi-VQA
Provides convenient interfaces for model inference
"""

import torch
from typing import Optional, List, Dict, Any


class VQAModelWrapper:
    """Base wrapper for VQA models"""
    
    def __init__(self, model, tokenizer, model_name: str, device: str = "cuda"):
        """
        Initialize model wrapper
        
        Args:
            model: Transformer model
            tokenizer: Tokenizer
            model_name: Model name/identifier
            device: Device to use ("cuda" or "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def to(self, device: str) -> None:
        """Move model to device"""
        self.device = device
        self.model.to(device)
        print(f"✅ Model moved to {device}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "dtype": self.model.dtype,
        }
    
    def print_model_info(self) -> None:
        """Print model information"""
        info = self.get_model_info()
        
        print(f"\n{'='*60}")
        print("Model Information")
        print(f"{'='*60}")
        print(f"Name: {info['model_name']}")
        print(f"Device: {info['device']}")
        print(f"Total Parameters: {info['num_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Dtype: {info['dtype']}")
        print(f"{'='*60}\n")


class VisionLanguageModelWrapper(VQAModelWrapper):
    """Wrapper for Vision-Language models (7B, 2B)"""
    
    def generate(
        self,
        images: List,
        questions: List[str],
        max_new_tokens: int = 48,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        Generate answers for VQA questions
        
        Args:
            images: List of PIL Images
            questions: List of question strings
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated answers
        """
        
        if not isinstance(images, list):
            images = [images]
        
        if not isinstance(questions, list):
            questions = [questions]
        
        # Build messages for each question
        messages_list = []
        for question in questions:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            messages_list.append(messages)
        
        # Apply chat template
        texts = []
        for messages in messages_list:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
        
        # Tokenize
        inputs = self.tokenizer(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                **kwargs,
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        answers = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return answers
    
    def answer_question(
        self,
        image,
        question: str,
        max_new_tokens: int = 48,
    ) -> str:
        """
        Answer a single VQA question
        
        Args:
            image: PIL Image
            question: Question string
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        answers = self.generate(
            images=[image],
            questions=[question],
            max_new_tokens=max_new_tokens,
        )
        return answers[0]


class TeacherModel(VisionLanguageModelWrapper):
    """7B Teacher model wrapper (MinhQuy24/Qwen-7B-Vi-VQA)"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__(
            model,
            tokenizer,
            model_name="Qwen-7B-Vi-VQA (Teacher)",
            device=device,
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get teacher model info"""
        info = super().get_model_info()
        info["model_type"] = "Teacher (7B)"
        return info


class StudentModel(VisionLanguageModelWrapper):
    """2B Student model wrapper (MinhQuy24/Qwen-2B-ViVQA)"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__(
            model,
            tokenizer,
            model_name="Qwen-2B-ViVQA (Student)",
            device=device,
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get student model info"""
        info = super().get_model_info()
        info["model_type"] = "Student (2B)"
        return info


class ModelPair:
    """Wrapper for teacher-student model pair"""
    
    def __init__(self, teacher: TeacherModel, student: StudentModel):
        """
        Initialize model pair
        
        Args:
            teacher: Teacher model wrapper
            student: Student model wrapper
        """
        self.teacher = teacher
        self.student = student
    
    def compare_answers(
        self,
        image,
        question: str,
        max_new_tokens: int = 48,
    ) -> Dict[str, str]:
        """
        Compare teacher and student answers
        
        Args:
            image: PIL Image
            question: Question string
            max_new_tokens: Maximum tokens
            
        Returns:
            Dictionary with teacher and student answers
        """
        
        teacher_answer = self.teacher.answer_question(
            image,
            question,
            max_new_tokens,
        )
        
        student_answer = self.student.answer_question(
            image,
            question,
            max_new_tokens,
        )
        
        return {
            "question": question,
            "teacher": teacher_answer,
            "student": student_answer,
        }
    
    def compare_batch(
        self,
        images,
        questions: List[str],
        max_new_tokens: int = 48,
    ) -> List[Dict[str, str]]:
        """
        Compare teacher and student on batch of questions
        
        Args:
            images: List of PIL Images
            questions: List of questions
            max_new_tokens: Maximum tokens
            
        Returns:
            List of comparison results
        """
        
        teacher_answers = self.teacher.generate(
            images,
            questions,
            max_new_tokens,
        )
        
        student_answers = self.student.generate(
            images,
            questions,
            max_new_tokens,
        )
        
        comparisons = [
            {
                "question": q,
                "teacher": t_ans,
                "student": s_ans,
            }
            for q, t_ans, s_ans in zip(questions, teacher_answers, student_answers)
        ]
        
        return comparisons
    
    def print_model_info(self) -> None:
        """Print info for both models"""
        print(f"\n{'='*70}")
        print("Model Pair Information")
        print(f"{'='*70}\n")
        
        print("👨‍🏫 TEACHER MODEL (7B)")
        print("-" * 70)
        self.teacher.print_model_info()
        
        print("👩‍🎓 STUDENT MODEL (2B)")
        print("-" * 70)
        self.student.print_model_info()


if __name__ == "__main__":
    # Example usage (requires models to be loaded)
    from model_loader import VQAModelLoader
    
    # Load models
    loader = VQAModelLoader()
    teacher_model, teacher_tokenizer = loader.load_teacher_model()
    student_model, student_tokenizer = loader.load_student_model()
    
    # Create wrappers
    teacher = TeacherModel(teacher_model, teacher_tokenizer)
    student = StudentModel(student_model, student_tokenizer)
    
    # Create pair
    pair = ModelPair(teacher, student)
    pair.print_model_info()

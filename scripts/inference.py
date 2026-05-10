"""
Inference script - Test model đã train
Chạy: python scripts/inference.py --image test.jpg --question "Mô tả ảnh này"
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Vi-VQA Inference")
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    parser.add_argument("--question", "-q", required=True, help="Question in Vietnamese")
    parser.add_argument("--model", "-m", default="MinhQuy24/Qwen-2B-ViVQA", help="Model path or HuggingFace ID")
    parser.add_argument("--max_tokens", default=48, type=int, help="Max new tokens")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True
    )
    processor = Qwen2VLProcessor.from_pretrained(args.model)
    print("Model loaded!")

    # Inference
    import torch
    from src.inference.predict import predict

    print(f"\nImage: {args.image}")
    print(f"Question: {args.question}")
    print("-" * 40)

    answer = predict(args.image, args.question, model, processor, args.max_tokens)
    print(f"Answer: {answer}")

    return answer


if __name__ == "__main__":
    main()
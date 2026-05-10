"""
Inference Module
Common inference functions cho cả teacher và student model
"""
import torch

def load_model_and_processor(model_path, device="auto"):
    """
    Load model và processor từ checkpoint

    Args:
        model_path: Path tới model checkpoint
        device: Device để load model

    Returns:
        Tuple (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=True
    )
    processor = Qwen2VLProcessor.from_pretrained(model_path)

    return model, processor

def build_vqa_inputs(image_path, question, processor):
    """
    Build inputs cho VQA inference

    Args:
        image_path: Path tới ảnh
        question: Câu hỏi
        processor: Qwen2VL processor

    Returns:
        Tokenized inputs
    """
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        images=[image_path],
        text=[prompt],
        return_tensors="pt"
    ).to(processor.image_processor.vision_config.device if hasattr(processor, 'image_processor') else "cuda")

    return inputs, prompt

def generate_answer(model, inputs, processor, max_new_tokens=48, do_sample=False):
    """
    Generate answer từ model

    Args:
        model: Qwen2VL model
        inputs: Tokenized inputs
        processor: Qwen2VL processor
        max_new_tokens: Max tokens to generate
        do_sample: Whether to use sampling

    Returns:
        Generated answer string
    """
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

    generated_ids = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def predict(image_path, question, model, processor, max_new_tokens=48):
    """
    Main prediction function

    Args:
        image_path: Path tới ảnh
        question: Câu hỏi tiếng Việt
        model: Qwen2VL model
        processor: Qwen2VL processor
        max_new_tokens: Max tokens

    Returns:
        Câu trả lời dạng string
    """
    inputs, _ = build_vqa_inputs(image_path, question, processor)
    return generate_answer(model, inputs, processor, max_new_tokens)
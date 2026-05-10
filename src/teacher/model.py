"""
Teacher Model Module
Load và inference với Qwen2-VL-7B-Instruct đã fine-tune
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

def load_teacher_model(model_path, device="auto"):
    """Load teacher model từ checkpoint"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=True
    )
    return model

def load_teacher_processor(model_path):
    """Load processor của teacher model"""
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    return processor

def get_answer(image_path, question, model, processor, max_new_tokens=48):
    """
    Lấy câu trả lời từ VQA model

    Args:
        image_path: Đường dẫn tới ảnh
        question: Câu hỏi tiếng Việt
        model: Qwen2VL model
        processor: Qwen2VL processor
        max_new_tokens: Số token tối đa cho generation

    Returns:
        Câu trả lời dạng string
    """
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        images=[image_path],
        text=[prompt],
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
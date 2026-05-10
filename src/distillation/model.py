"""
Student Model Module
Load student model (Qwen2-VL-2B) với LoRA
"""
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model

def load_student_model(model_name="Qwen/Qwen2-VL-2B-Instruct", device="auto"):
    """Load student model với 4-bit quantization"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device,
        load_in_4bit=True
    )
    return model

def load_student_processor(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """Load student processor"""
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    return processor

def apply_lora_config(model, r=16, lora_alpha=16):
    """Apply LoRA config cho student model"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Tỷ lệ trainable: {100 * trainable_params / total_params:.2f}%")

    return model

def save_student_model(model, processor, save_dir):
    """Lưu student model và processor"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Đã lưu student model và processor vào {save_dir}")
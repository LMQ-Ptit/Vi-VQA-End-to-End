"""
Script train student model bằng Knowledge Distillation
Chạy: python scripts/train_distill.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
import yaml

from src.teacher.model import get_answer as teacher_answer
from src.teacher.dataset import load_vqa_split
from src.distillation.model import apply_lora_config, save_student_model

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "student.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("="*50)
    print("Knowledge Distillation (7B → 2B)")
    print("="*50)

    # Load teacher model
    print("\n[1/6] Loading teacher model...")
    teacher_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config["teacher"]["path"],
        device_map="auto",
        load_in_4bit=True
    )
    teacher_processor = Qwen2VLProcessor.from_pretrained(config["teacher"]["path"])
    print("Teacher loaded!")

    # Load student model
    print("\n[2/6] Loading student model...")
    student_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config["model"]["name"],
        device_map="auto",
        load_in_4bit=True
    )
    student_processor = Qwen2VLProcessor.from_pretrained(config["model"]["name"])
    print("Student loaded!")

    # Apply LoRA
    print("\n[3/6] Applying LoRA to student...")
    student_model = apply_lora_config(
        student_model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"]
    )

    # Load data
    print("\n[4/6] Loading dataset...")
    train_data = load_vqa_split(
        config["data"]["train_path"],
        config["data"]["train_images"]
    )
    train_dataset = _VQADataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    print(f"Loaded {len(train_dataset)} samples")

    # Setup optimizer
    print("\n[5/6] Setting up optimizer...")
    optimizer = AdamW8bit(student_model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    print("\n[6/6] Starting distillation...")
    _distillation_loop(
        student_model, teacher_model,
        student_processor, teacher_processor,
        train_loader, optimizer,
        num_epochs=config["training"]["num_epochs"],
        max_new_tokens=config["teacher"]["max_new_tokens"]
    )

    # Save
    save_dir = "models/student"
    os.makedirs(save_dir, exist_ok=True)
    save_student_model(student_model, student_processor, save_dir)

    print("\n" + "="*50)
    print("Distillation completed!")
    print(f"Model saved to: {save_dir}")
    print("="*50)


class _VQADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)


def _distillation_loop(student_model, teacher_model, student_processor, teacher_processor, train_loader, optimizer, num_epochs=1, max_new_tokens=48):
    """Training loop cho distillation"""
    import torch.nn.functional as F
    from tqdm import tqdm

    student_model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            image_path = batch["image_path"][0]
            question = batch["question"][0]

            # 1. Get teacher answer
            with torch.no_grad():
                t_answer = teacher_answer(image_path, question, teacher_model, teacher_processor, max_new_tokens)

            # 2. Create full text
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            prompt = student_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt + t_answer

            # 3. Tokenize
            inputs = student_processor(
                images=[image_path],
                text=[full_text],
                return_tensors="pt",
                padding="max_length",
                max_length=512
            ).to(student_model.device)

            # 4. Mask prompt
            labels = inputs["input_ids"].clone()
            prompt_len = len(student_processor.tokenizer(prompt)["input_ids"])
            labels[0, :prompt_len] = -100

            # 5. Get logits
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.detach()

            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits

            # 6. KL loss
            loss = _compute_kl_loss(teacher_logits, student_logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})


def _compute_kl_loss(teacher_logits, student_logits, labels, temperature=1.0):
    import torch.nn.functional as F

    mask = (labels != -100)
    t_logits = teacher_logits[mask]
    s_logits = student_logits[mask]

    min_vocab = min(t_logits.shape[-1], s_logits.shape[-1])
    t_logits = t_logits[..., :min_vocab]
    s_logits = s_logits[..., :min_vocab]

    min_len = min(t_logits.shape[0], s_logits.shape[0])
    t_logits = t_logits[:min_len, :]
    s_logits = s_logits[:min_len, :]

    loss = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=-1),
        F.softmax(t_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    return loss


if __name__ == "__main__":
    main()
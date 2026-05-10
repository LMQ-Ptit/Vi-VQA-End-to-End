"""
Distillation Trainer Module
Training loop cho knowledge distillation (7B → 2B)
"""
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from bitsandbytes.optim import AdamW8bit

def create_distillation_optimizer(model, lr=2e-5):
    """Tạo optimizer cho distillation"""
    return AdamW8bit(model.parameters(), lr=lr)

def distillation_training_loop(
    student_model,
    teacher_model,
    student_processor,
    train_loader,
    optimizer,
    num_epochs=1,
    max_new_tokens=48
):
    """
    Training loop cho knowledge distillation

    Args:
        student_model: Student model (2B)
        teacher_model: Teacher model (7B)
        student_processor: Processor cho student
        train_loader: DataLoader
        optimizer: AdamW8bit optimizer
        num_epochs: Số epoch
        max_new_tokens: Max tokens cho teacher inference

    Returns:
        Dictionary với training metrics
    """
    student_model.train()
    optimizer.zero_grad()

    total_loss = 0
    steps = 0

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            image_path = batch["image_path"][0]
            question = batch["question"][0]

            # 1. Sinh answer từ teacher
            with torch.no_grad():
                teacher_answer = _get_teacher_answer(
                    image_path, question, teacher_model, teacher_processor, max_new_tokens
                )

            # 2. Tạo full text từ student prompt + teacher answer
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            prompt = student_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + teacher_answer

            # 3. Tokenize input
            inputs = student_processor(
                images=[image_path],
                text=[full_text],
                return_tensors="pt",
                padding="max_length",
                max_length=512
            ).to(student_model.device)

            # 4. Mask prompt tokens
            labels = inputs["input_ids"].clone()
            prompt_len = len(student_processor.tokenizer(prompt)["input_ids"])
            labels[0, :prompt_len] = -100

            # 5. Get logits từ teacher và student
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.detach()

            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits

            # 6. Tính KL divergence loss
            loss = _compute_kl_loss(
                teacher_logits, student_logits, labels
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": loss.item()})

    return {
        "avg_loss": total_loss / steps if steps > 0 else 0,
        "total_steps": steps
    }


def _get_teacher_answer(image_path, question, model, processor, max_new_tokens):
    """Helper: get answer từ teacher model"""
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


def _compute_kl_loss(teacher_logits, student_logits, labels, temperature=1.0):
    """Compute KL divergence loss giữa teacher và student logits"""
    mask = (labels != -100)
    t_logits = teacher_logits[mask]
    s_logits = student_logits[mask]

    # Fix vocab size mismatch
    min_vocab = min(t_logits.shape[-1], s_logits.shape[-1])
    t_logits = t_logits[..., :min_vocab]
    s_logits = s_logits[..., :min_vocab]

    # Fix sequence length mismatch
    min_len = min(t_logits.shape[0], s_logits.shape[0])
    t_logits = t_logits[:min_len, :]
    s_logits = s_logits[:min_len, :]

    # KL Divergence loss
    loss = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=-1),
        F.softmax(t_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    return loss
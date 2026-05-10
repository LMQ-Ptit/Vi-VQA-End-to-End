"""
Teacher Trainer Module
Training logic cho teacher model (7B)
"""
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from datasets import Dataset

def load_model_and_tokenizer(model_name="unsloth/Qwen2-VL-7B-Instruct", max_seq_length=2048):
    """Load teacher model với LoRA config"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Giảm số visual tokens để tránh OOM
    tokenizer.image_processor.max_pixels = 256 * 28 * 28
    tokenizer.image_processor.min_pixels = 128 * 28 * 28

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    return model, tokenizer

def create_vqa_trainer(model, train_dataset, eval_dataset=None, tokenizer=None):
    """Tạo SFTTrainer cho VQA task"""
    model.config.use_cache = False

    if tokenizer is None:
        raise ValueError("tokenizer is required for SFTTrainer")

    if hasattr(tokenizer, "image_processor") and tokenizer.image_processor is not None:
        tokenizer.image_processor.max_pixels = 256 * 28 * 28
        tokenizer.image_processor.min_pixels = 128 * 28 * 28

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=1e-5,
            bf16=True,
            fp16=False,
            optim="adamw_8bit",
            logging_steps=1,
            output_dir="outputs-7B",
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            remove_unused_columns=False,
        ),
    )

    trainer.data_collator = DataCollatorForVisionLanguageModeling(
        processor=tokenizer,
        max_length=1024,
    )

    return trainer

def train_with_checkpoint(trainer):
    """Train với khả năng resume từ checkpoint"""
    import os

    output_dir = trainer.args.output_dir
    checkpoint_dirs = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ] if os.path.exists(output_dir) else []

    if checkpoint_dirs:
        latest_checkpoint = max(
            checkpoint_dirs,
            key=lambda x: int(x.split("-")[-1])
        )
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(f"Resume từ checkpoint: {checkpoint_path}")
        return trainer.train(resume_from_checkpoint=checkpoint_path)

    print("Không tìm thấy checkpoint, bắt đầu train mới")
    return trainer.train()

def save_final_model(model, tokenizer, save_dir):
    """Lưu model và tokenizer cuối cùng"""
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Đã lưu model và tokenizer vào: {save_dir}")
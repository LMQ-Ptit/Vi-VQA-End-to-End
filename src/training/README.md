# Training Module - Vi-VQA

Training utilities for fine-tuning Vision-Language models on Vietnamese VQA task.

## Overview

This module provides:
- **VQATrainer** - Train base model on VQA dataset
- **DistillationTrainer** - Train student model via Knowledge Distillation
- **Configuration utilities** - Load and manage YAML configs
- **Training callbacks & metrics** - Monitor and evaluate training

## Files

| File | Purpose |
|------|---------|
| `config_utils.py` | Load and convert YAML configs to training arguments |
| `trainer_utils.py` | Callbacks, metrics, and training helpers |
| `training_loop.py` | Main training loop (VQATrainer & DistillationTrainer) |
| `__init__.py` | Module exports |

## Usage

### 1. Train Base Model (7B)

```python
import sys
sys.path.insert(0, '../')

from models import VQAModelLoader, TeacherModel
from data import VQADataLoader, prepare_vqa_dataset
from training import VQATrainer, TrainingHelper

# Print GPU info
TrainingHelper.print_gpu_info()

# Load model
loader = VQAModelLoader()
model, tokenizer = loader.load_base_model()

# Load dataset
data_loader = VQADataLoader()
train_dataset = data_loader.get_train()
dev_dataset = data_loader.get_dev()

# Prepare datasets
train_prepared = prepare_vqa_dataset(train_dataset, tokenizer, max_samples=1000)
dev_prepared = prepare_vqa_dataset(dev_dataset, tokenizer, max_samples=200)

# Create trainer
trainer = VQATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_prepared,
    eval_dataset=dev_prepared,
    config_dir="../../configs",
    output_dir="../../outputs-7B",
)

# Setup and train
trainer.setup_trainer()
train_result = trainer.train(resume_from_checkpoint=True)

# Evaluate
eval_result = trainer.evaluate()

# Save model
trainer.save_model()
trainer.save_tokenizer()
```

### 2. Knowledge Distillation (7B -> 2B)

```python
from models import VQAModelLoader
from data import VQADataLoader, prepare_vqa_dataset
from training import DistillationTrainer

# Load models
loader = VQAModelLoader()
teacher_m, teacher_t = loader.load_teacher_model()
student_m, student_t = loader.load_student_model()

# Load and prepare dataset
data_loader = VQADataLoader()
train_dataset = data_loader.get_train()
dev_dataset = data_loader.get_dev()

train_prepared = prepare_vqa_dataset(train_dataset, student_t, max_samples=1000)
dev_prepared = prepare_vqa_dataset(dev_dataset, student_t, max_samples=200)

# Create distillation trainer
distillation_trainer = DistillationTrainer(
    student_model=student_m,
    student_tokenizer=student_t,
    teacher_model=teacher_m,
    teacher_tokenizer=teacher_t,
    train_dataset=train_prepared,
    eval_dataset=dev_prepared,
    config_dir="../../configs",
    output_dir="../../outputs-2B-distilled",
)

# Train
distillation_trainer.setup_trainer()
train_result = distillation_trainer.train()

# Save student model
distillation_trainer.save_student_model()
```

## Configuration

Training is configured via YAML files in `configs/`:

- **training_config.yaml** - Training hyperparameters
- **distillation_config.yaml** - Distillation parameters
- **model_config.yaml** - Model specifications

### Key Parameters

**training_config.yaml:**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  learning_rate: 1e-5
  warmup_steps: 10
  save_steps: 200
```

**distillation_config.yaml:**
```yaml
distillation:
  temperature: 4.0       # Softness of targets
  alpha: 0.5            # KD loss weight (0.5 = equal with task loss)
  training:
    num_train_epochs: 2
    learning_rate: 5e-5
```

## Classes & APIs

### ConfigLoader

```python
from training import ConfigLoader

loader = ConfigLoader(config_dir="./configs")

# Load individual configs
training_config = loader.get_training_config()
model_config = loader.get_model_config()
data_config = loader.get_data_config()

# Convert to training args
training_args = ConfigLoader.create_training_args(training_config)
lora_config = ConfigLoader.create_lora_config(model_config)

# Setup environment variables
ConfigLoader.setup_environment(training_config)
```

### TrainingHelper

```python
from training import TrainingHelper

# Print GPU info
TrainingHelper.print_gpu_info()

# Get device
device = TrainingHelper.get_device()

# Get GPU memory usage
memory = TrainingHelper.get_gpu_memory_usage()
```

### MetricsCalculator

```python
from training import MetricsCalculator

predictions = ["answer1", "answer2", "answer3"]
references = ["ref1", "ref2", "ref3"]

# Individual metrics
exact_match = MetricsCalculator.exact_match(predictions, references)
rouge_l = MetricsCalculator.rouge_l(predictions, references)
bertscore = MetricsCalculator.bertscore_f1(predictions, references)

# All metrics at once
metrics = MetricsCalculator.calculate_all_metrics(predictions, references)
# Output: {"exact_match": 0.188, "rouge_l": 0.680, "bertscore_f1": 0.874}
```

### CheckpointManager

```python
from training import CheckpointManager

# Find latest checkpoint
latest = CheckpointManager.find_latest_checkpoint("./outputs-7B")
# Returns: "./outputs-7B/checkpoint-1000"

# Resume from checkpoint (automatic in VQATrainer)
CheckpointManager.resume_from_checkpoint(trainer, "./outputs-7B")
```

### VQATrainer

```python
from training import VQATrainer

trainer = VQATrainer(model, tokenizer, train_dataset, eval_dataset)

# Setup (required)
trainer.setup_trainer()

# Train (with automatic checkpoint resume)
train_result = trainer.train(resume_from_checkpoint=True)

# Evaluate
eval_result = trainer.evaluate()

# Save
trainer.save_model("./outputs-7B/final_lora")
trainer.save_tokenizer("./outputs-7B/final_lora")

# Info
trainer.print_training_summary()
```

### DistillationTrainer

```python
from training import DistillationTrainer

distill_trainer = DistillationTrainer(
    student_model, student_tokenizer,
    teacher_model, teacher_tokenizer,
    train_dataset, eval_dataset
)

# Train
distill_trainer.setup_trainer()
distill_trainer.train()

# Save
distill_trainer.save_student_model("./outputs-2B/final_lora")
```

## Training Workflow

### Option 1: Train from Scratch (7B)

```
1. Load base model (unsloth/Qwen2-VL-7B-Instruct)
2. Load VQA dataset (MinhQuy24/vlsp2023-vqa-dataset)
3. Setup VQATrainer
4. Train for N epochs
5. Save as "outputs-7B/final_lora"
6. Push to HuggingFace (MinhQuy24/Qwen-7B-Vi-VQA)
```

### Option 2: Knowledge Distillation (7B -> 2B)

```
1. Load teacher (MinhQuy24/Qwen-7B-Vi-VQA)
2. Load student (Qwen/Qwen2-VL-2B-Instruct)
3. Load VQA dataset
4. Setup DistillationTrainer
5. Train student with KD loss
6. Save as "outputs-2B/final_lora"
7. Push to HuggingFace (MinhQuy24/Qwen-2B-ViVQA)
```

## Metrics

Supports multiple evaluation metrics:

- **Exact Match** - % of exact string matches
- **ROUGE-L** - F1 score based on longest common subsequence
- **BERTScore F1** - Contextual semantic similarity

```python
metrics = MetricsCalculator.calculate_all_metrics(predictions, references)
# {"exact_match": 0.188, "rouge_l": 0.680, "bertscore_f1": 0.874}
```

## Callbacks

`VQATrainerCallback` automatically logs:
- Training loss per step
- Evaluation loss per eval
- Training history saved to `training_history.json`

## Advanced: Custom Training Loop

```python
from training import VQATrainer
from transformers import TrainingArguments

trainer = VQATrainer(model, tokenizer, train_dataset)

# Customize before setup
trainer.output_dir = "./custom_output"

# Setup
trainer.setup_trainer()

# Access underlying trainer
base_trainer = trainer.get_trainer()

# Custom training
base_trainer.train()
```

## Environment Variables

Automatically set from config:

```yaml
environment:
  pytorch_cuda_alloc_conf: "expandable_segments:True"
  hf_hub_connect_timeout: 60
  hf_hub_read_timeout: 60
  hf_hub_disable_telemetry: true
```

## Troubleshooting

**Q: Out of Memory**
- Reduce `per_device_train_batch_size` (default: 1)
- Increase `gradient_accumulation_steps`
- Use `load_in_4bit=True` for models

**Q: Training too slow**
- Increase batch size (if memory allows)
- Reduce number of epochs
- Use fewer samples (`max_samples` in prepare_vqa_dataset)

**Q: Loss not decreasing**
- Try different learning rate (default: 1e-5)
- Check data preprocessing
- Try more warmup steps

## Example Training Script

See `scripts/train.py` for full working example.

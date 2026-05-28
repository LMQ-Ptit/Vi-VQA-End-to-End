# Scripts - Vi-VQA Executable Entry Points

Standalone executable scripts for the entire Vi-VQA pipeline:
- **Training** (7B on VQA)
- **Distillation** (7B to 2B)
- **Inference** (single, batch, comparison)
- **Evaluation** (metrics calculation)

## Quick Start

### 1. Train Base Model (7B)

```bash
cd scripts
python train.py --config-dir ../configs --output-dir ../outputs-7B
```

**Features:**
- Loads base model: `unsloth/Qwen2-VL-7B-Instruct`
- Loads dataset: `MinhQuy24/vlsp2023-vqa-dataset`
- Applies LoRA fine-tuning
- Auto-resumes from checkpoint
- Evaluates on dev set
- Saves trained model locally
- Optional: Push to HuggingFace Hub

**Output:**
```
../outputs-7B/
├── checkpoint-1000/
├── checkpoint-2000/
├── final_lora/          # Final trained model
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── special_tokens_map.json
└── training_history.json
```

### 2. Knowledge Distillation (7B to 2B)

```bash
python distill.py --config-dir ../configs --output-dir ../outputs-2B-distilled
```

**Features:**
- Loads teacher: `MinhQuy24/Qwen-7B-Vi-VQA` (7B)
- Loads student: `Qwen/Qwen2-VL-2B-Instruct` (2B)
- Knowledge Distillation training:
  - Temperature: 4.0 (soft targets)
  - Alpha: 0.5 (KD loss weight)
  - Method: Response-based
- Auto-resumes from checkpoint
- Saves distilled student model
- Optional: Push to HuggingFace Hub

**Output:**
```
../outputs-2B-distilled/
├── checkpoint-500/
├── checkpoint-1000/
├── final_lora/          # Distilled student model
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── special_tokens_map.json
└── training_history.json
```

### 3. Run Inference

#### Single Image Inference
```bash
python inference.py \
  --model-type teacher \
  --image-path /path/to/image.jpg \
  --question "Đây là gì?"
```

Output:
```
Image: /path/to/image.jpg
Question: Đây là gì?
Answer: Một chú mèo
Confidence: 0.95
```

#### Batch Inference from JSON
```bash
python inference.py \
  --model-type teacher \
  --input-json test_data.json \
  --output-json predictions.json
```

**Input JSON format:**
```json
[
  {
    "id": 1,
    "image_path": "/path/to/image1.jpg",
    "question": "Có bao nhiêu mèo?"
  },
  {
    "id": 2,
    "image_path": "/path/to/image2.jpg",
    "question": "Đây là gì?"
  }
]
```

**Output JSON:**
```json
[
  {
    "id": 1,
    "prediction": "Hai chú mèo",
    "confidence": 0.92
  },
  {
    "id": 2,
    "prediction": "Một chiếc ghế",
    "confidence": 0.88
  }
]
```

#### Compare Teacher vs Student
```bash
python inference.py \
  --model-type both \
  --input-json test_data.json \
  --output-json comparison.json
```

**Output JSON:**
```json
[
  {
    "id": 1,
    "question": "Có bao nhiêu mèo?",
    "teacher_answer": "Hai chú mèo",
    "student_answer": "Hai mèo",
    "match": true
  }
]
```

### 4. Evaluate on Test Set

#### Evaluate Single Model
```bash
python evaluate.py \
  --model-type teacher \
  --output-file eval_results.json
```

**Output:**
```
Evaluation Results
==================
Model: TEACHER
Samples: 1000

Metrics:
  Exact Match: 85.20%
  Partial Match: 0.9234
  BLEU Score: 0.8745
```

#### Compare Teacher vs Student
```bash
python evaluate.py \
  --model-type both \
  --output-file comparison.json
```

**Output:**
```
Comparison Results
==================
Samples: 1000

Teacher (7B) Metrics:
  Exact Match: 85.20%
  Partial Match: 0.9234
  BLEU Score: 0.8745

Student (2B) Metrics:
  Exact Match: 82.10%
  Partial Match: 0.9012
  BLEU Score: 0.8512

Teacher-Student Agreement: 950/1000 (95.0%)
```

---

## Scripts Reference

### train.py

**Purpose:** Train 7B base model on Vietnamese VQA dataset

**Usage:**
```bash
python train.py [options]
```

**Options:**
```
--config-dir DIR          Path to configs directory (default: ./configs)
--output-dir DIR          Output directory (default: ./outputs-7B)
--resume                  Resume from latest checkpoint
--max-train-samples N     Max training samples (default: all)
--max-eval-samples N      Max eval samples (default: all)
--push-to-hub             Push to HuggingFace Hub
--hub-model-id ID         Hub model ID (default: MinhQuy24/Qwen-7B-Vi-VQA)
```

**Workflow:**
```
1. Print GPU info
2. Load base model (unsloth/Qwen2-VL-7B-Instruct)
3. Load VQA dataset
4. Prepare datasets (tokenization, formatting)
5. Setup SFTTrainer with LoRA
6. Train with checkpoint resume support
7. Evaluate on dev set
8. Save model locally
9. [Optional] Push to HuggingFace
```

### distill.py

**Purpose:** Knowledge distillation from 7B teacher to 2B student

**Usage:**
```bash
python distill.py [options]
```

**Options:**
```
--config-dir DIR          Path to configs directory (default: ./configs)
--output-dir DIR          Output directory (default: ./outputs-2B-distilled)
--resume                  Resume from latest checkpoint
--max-train-samples N     Max training samples (default: all)
--max-eval-samples N      Max eval samples (default: all)
--push-to-hub             Push to HuggingFace Hub
--hub-model-id ID         Hub model ID (default: MinhQuy24/Qwen-2B-ViVQA)
```

**KD Parameters:**
- Temperature: 4.0 (from distillation_config.yaml)
- Alpha: 0.5 (KD loss weight)
- Method: Response-based

**Workflow:**
```
1. Print GPU info
2. Load teacher (7B trained model)
3. Load student (2B base model)
4. Load VQA dataset
5. Prepare datasets
6. Setup DistillationTrainer with KD loss
7. Train student with knowledge distillation
8. Evaluate student on dev set
9. Save student model locally
10. [Optional] Push to HuggingFace
```

### inference.py

**Purpose:** Run inference with trained models

**Usage:**
```bash
python inference.py [options]
```

**Options:**
```
--model-type TYPE         Model: teacher, student, or both (default: teacher)
--image-path PATH         Image path (for single inference)
--question TEXT           Question (for single inference)
--input-json FILE         Input JSON file (for batch inference)
--output-json FILE        Output JSON file (for saving results)
```

**Modes:**

| Mode | Usage | Command |
|------|-------|---------|
| Single (Teacher) | One image + question | `--image-path IMG --question Q` |
| Single (Student) | One image + question | `--model-type student --image-path IMG --question Q` |
| Batch (Teacher) | JSON file | `--input-json data.json --output-json pred.json` |
| Batch (Student) | JSON file | `--model-type student --input-json data.json` |
| Compare | JSON file | `--model-type both --input-json data.json` |

### evaluate.py

**Purpose:** Evaluate models on test set

**Usage:**
```bash
python evaluate.py [options]
```

**Options:**
```
--model-type TYPE         Model: teacher, student, or both (default: teacher)
--test-json FILE          Custom test JSON file (default: HuggingFace test set)
--output-file FILE        Save results to JSON file
```

**Metrics:**
- **Exact Match**: Percentage of exact string matches
- **ROUGE-L**: F1 score based on longest common subsequence
- **BERTScore F1**: Contextual semantic similarity using BERT embeddings

**Modes:**

| Mode | Command |
|------|---------|
| Evaluate Teacher | `python evaluate.py --model-type teacher` |
| Evaluate Student | `python evaluate.py --model-type student` |
| Compare Both | `python evaluate.py --model-type both` |
| With Custom Test | `python evaluate.py --test-json custom.json` |

---

## Input/Output Formats

### JSON Format for Batch Operations

**Input (test_data.json):**
```json
[
  {
    "id": 1,
    "image_path": "/path/to/image.jpg",
    "question": "Câu hỏi tiếng Việt?",
    "answer": "Đáp án đúng"  # For evaluation
  }
]
```

**Inference Output (predictions.json):**
```json
[
  {
    "id": 1,
    "prediction": "Dự đoán của mô hình",
    "confidence": 0.85
  }
]
```

**Evaluation Output (eval_results.json):**
```json
{
  "model": "teacher",
  "num_samples": 1000,
  "metrics": {
    "exact_match": 0.188,
    "rouge_l": 0.680,
    "bertscore_f1": 0.874
  },
  "predictions": ["answer1", "answer2", ...],
  "references": ["ref1", "ref2", ...]
}
```

---

## Complete Training Pipeline

### Step 1: Train 7B Model
```bash
python train.py \
  --config-dir ../configs \
  --output-dir ../outputs-7B \
  --max-train-samples 5000 \
  --max-eval-samples 1000
```

### Step 2: Push to HuggingFace (Manual)
```bash
huggingface-cli upload MinhQuy24/Qwen-7B-Vi-VQA \
  ../outputs-7B/final_lora
```

### Step 3: Knowledge Distillation
```bash
python distill.py \
  --config-dir ../configs \
  --output-dir ../outputs-2B-distilled \
  --max-train-samples 5000
```

### Step 4: Evaluate Both Models
```bash
python evaluate.py --model-type both --output-file comparison.json
```

### Step 5: Run Inference
```bash
python inference.py \
  --model-type both \
  --input-json test_samples.json \
  --output-json results.json
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size in training_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Increase
```

### Slow Training
- Reduce dataset size: `--max-train-samples 1000`
- Reduce epochs in config files
- Use A100 GPU if available

### Model Not Loading
- Ensure HuggingFace credentials are set: `huggingface-cli login`
- Check internet connection
- Verify model exists on HuggingFace Hub

### Inference Not Working
- Verify image paths in JSON are correct
- Ensure image format is supported (JPG, PNG)
- Check question is in Vietnamese (or modify code for other languages)

---

## Configuration Files

All scripts use YAML configs in `../configs/`:

| File | Purpose |
|------|---------|
| `training_config.yaml` | Training hyperparameters |
| `distillation_config.yaml` | KD parameters |
| `model_config.yaml` | Model specifications |
| `data_config.yaml` | Dataset paths |
| `config.yaml` | Main pipeline config |

---

## Example Workflows

### Minimal Training (Debug)
```bash
python train.py \
  --max-train-samples 100 \
  --max-eval-samples 50 \
  --output-dir ../outputs-7B-debug
```

### Full Training Pipeline
```bash
# 1. Train
python train.py --config-dir ../configs --output-dir ../outputs-7B --resume

# 2. Evaluate
python evaluate.py --model-type teacher --output-file eval_7b.json

# 3. Distill
python distill.py --config-dir ../configs --output-dir ../outputs-2B-distilled --resume

# 4. Compare
python evaluate.py --model-type both --output-file comparison.json

# 5. Inference
python inference.py --model-type teacher --input-json samples.json --output-json pred.json
```

### Production Deployment
```bash
# Train with all data
python train.py --push-to-hub --hub-model-id YourName/Qwen-7B-Vi-VQA

# Distill
python distill.py --push-to-hub --hub-model-id YourName/Qwen-2B-ViVQA

# Create evaluation report
python evaluate.py --model-type both --output-file metrics_report.json
```

---

## Dependencies

All scripts use modules from `src/`:
- `src.models` - Model loading and wrapping
- `src.data` - Data loading and preparation
- `src.training` - Training utilities and main trainers

Install dependencies:
```bash
pip install -r ../requirements.txt
```

---

## Logging

All scripts output detailed logs with timestamps:
- Step-by-step progress
- GPU information
- Dataset loading status
- Training time and loss
- Metrics and results

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Tips & Best Practices

Always use `--resume` when training to avoid losing progress
Test with small dataset first using `--max-train-samples 100`
Save evaluation results with `--output-file` for comparison
Monitor GPU usage - scripts print GPU info at start
Use batch inference for speed - faster than single samples
Compare teacher vs student to validate distillation quality
Keep configs in YAML - easy to track and reproduce experiments


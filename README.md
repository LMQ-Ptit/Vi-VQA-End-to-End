# Vi-VQA - Vietnamese Visual Question Answering

Vietnamese VQA project using Knowledge Distillation from Qwen2-VL-7B (Teacher) to Qwen2-VL-2B (Student).

## Project Structure

```
Pine_line/
├── configs/                    # YAML configurations
│   ├── teacher.yaml            # Teacher 7B training config
│   └── student.yaml            # Student 2B distillation config
├── notebooks/                  # Original source notebooks (reference)
├── scripts/                    # Python scripts
│   ├── train_teacher.py        # Train teacher model
│   ├── train_distill.py        # Knowledge distillation
│   ├── inference.py            # Test inference
│   └── run_api.py              # Run API + Gradio UI
├── src/                        # Code modules
│   ├── teacher/                # Teacher model utils
│   ├── distillation/           # Distillation utils
│   ├── inference/              # Common inference
│   └── api/                    # API & UI
├── models/                     # Trained models storage
├── outputs/                    # Checkpoints
└── README.md
```

## Hardware Requirements

| Task | VRAM | GPU |
|------|------|-----|
| Train Teacher (7B) | ~22GB | L4, A10G, A100 |
| Train Student (2B) | ~12GB | T4, L4 |
| Inference | ~8GB | T4, L4 |

## Installation

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
# Install unsloth (auto-installs required packages)
pip install unsloth

# Or use requirements.txt
pip install -r requirements.txt
```

## Usage

### 1. Train Teacher Model (7B)

Train 7B model from pretrained Qwen2-VL-7B-Instruct.

```bash
python scripts/train_teacher.py
```

**Output**: `models/teacher/` contains LoRA weights and tokenizer

---

### 2. Knowledge Distillation (7B → 2B)

Distill knowledge from teacher 7B to student 2B.

```bash
python scripts/train_distill.py
```

**Before running**: Edit `configs/student.yaml` to point to teacher checkpoint:

```yaml
teacher:
  path: "models/teacher"
```

**Output**: `models/student/`

---

### 3. Inference (Quick Test)

Test trained model with an image and question.

```bash
# Use pretrained model from HuggingFace
python scripts/inference.py -i test.jpg -q "Describe the image"

# Use trained model
python scripts/inference.py -i test.jpg -q "Describe the image" -m models/student
```

---

### 4. Run API Server + Gradio UI

Run FastAPI backend + Gradio frontend.

```bash
python scripts/run_api.py
```

**Access:**
- Gradio UI: http://localhost:7860
- API Docs: http://localhost:8000/docs

### Demo Interface

![Vi-VQA Interface](image.png)

---

## Complete Training Pipeline

```bash
# Step 1: Train Teacher
python scripts/train_teacher.py

# Step 2: Distill to Student
python scripts/train_distill.py

# Step 3: Test inference
python scripts/inference.py -i test.jpg -q "Describe the image"

# Step 4: Deploy API
python scripts/run_api.py
```


## Common Troubleshooting

### OOM (Out of Memory) Error
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`

### Truncation error during training
```yaml
# In configs/teacher.yaml
image_processor:
  max_pixels: 256 * 28 * 28
  min_pixels: 128 * 28 * 28
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_teacher.py` | Train teacher model 7B |
| `scripts/train_distill.py` | Knowledge distillation 7B→2B |
| `scripts/inference.py` | Single inference test |
| `scripts/run_api.py` | Run FastAPI + Gradio UI |

## Modules

| Module | Description |
|--------|-------------|
| `src/teacher/` | Load & train teacher model |
| `src/distillation/` | Student model & distillation loop |
| `src/inference/` | Common inference functions |
| `src/api/` | FastAPI backend & Gradio frontend |
# Model Management - Vi-VQA

Model loading and inference utilities for Vi-VQA project.

## Available Models

| Model | Size | Purpose | HuggingFace |
|-------|------|---------|------------|
| Qwen-7B-Vi-VQA | 7B | Teacher (trained) | `MinhQuy24/Qwen-7B-Vi-VQA` |
| Qwen-2B-ViVQA | 2B | Student (distilled) | `MinhQuy24/Qwen-2B-ViVQA` |
| Qwen2-VL-7B-Instruct | 7B | Base (for training) | `unsloth/Qwen2-VL-7B-Instruct` |

## Module Overview

### `model_loader.py` - Load Models from HuggingFace

```python
from src.models import VQAModelLoader

# Initialize loader
loader = VQAModelLoader(device="cuda")

# Load teacher model (7B trained)
teacher_model, teacher_tokenizer = loader.load_teacher_model()

# Load student model (2B distilled)
student_model, student_tokenizer = loader.load_student_model()

# Load base model (for training from scratch)
base_model, base_tokenizer = loader.load_base_model()

# List available models
VQAModelLoader.print_available_models()
```

**Key Features:**
- Auto-download from HuggingFace Hub
- Support 4-bit/8-bit quantization
- Memory-efficient loading
- Token-based authentication

### `model_wrapper.py` - Model Wrapper Classes

```python
from src.models import TeacherModel, StudentModel, ModelPair

# Create wrappers
teacher = TeacherModel(teacher_model, teacher_tokenizer)
student = StudentModel(student_model, student_tokenizer)

# Single prediction
answer = teacher.answer_question(image, question)

# Batch predictions
answers = teacher.generate(images, questions)

# Compare models
pair = ModelPair(teacher, student)
comparison = pair.compare_answers(image, question)
```

**Classes:**
- `VQAModelWrapper` - Base wrapper
- `VisionLanguageModelWrapper` - VL model wrapper with generation
- `TeacherModel` - 7B teacher wrapper
- `StudentModel` - 2B student wrapper
- `ModelPair` - Teacher-student comparison

### `inference.py` - Inference Utilities

```python
from src.models import VQAInference, ComparativeInference

# Single model inference
inference = VQAInference(model_wrapper)
answer = inference.predict_single(image, question)

# Batch inference
answers = inference.predict_batch(images, questions)

# From file paths
answer = inference.predict_from_file("path/to/image.jpg", question)

# From JSON dataset
predictions = inference.predict_from_json("data.json", limit=100)
inference.save_predictions(predictions, "output.json")

# Evaluate predictions
score = VQAInference.evaluate_predictions(predictions, metric="exact_match")

# Compare teacher vs student
comparator = ComparativeInference(teacher_wrapper, student_wrapper)
comparison = comparator.compare_single(image, question)
comparator.save_comparison(comparisons, "comparison.json")
```

**Features:**
- Single and batch inference
- Load from files or JSON
- Save predictions
- Evaluate metrics (exact match, partial match)
- Teacher-student comparison

## Usage Examples

### Example 1: Load and Use Teacher Model

```python
import sys
sys.path.insert(0, '../')

from models import VQAModelLoader, TeacherModel, VQAInference
from PIL import Image

# Load model
loader = VQAModelLoader()
model, tokenizer = loader.load_teacher_model()

# Create wrapper and inference
teacher = TeacherModel(model, tokenizer)
inference = VQAInference(teacher)

# Predict
image = Image.open("test.jpg")
question = "Hãy mô tả bức ảnh này"
answer = inference.predict_single(image, question)

print(f"Q: {question}")
print(f"A: {answer}")
```

### Example 2: Compare Teacher and Student

```python
from models import VQAModelLoader, TeacherModel, StudentModel, ModelPair
from models import ComparativeInference

# Load both models
loader = VQAModelLoader()
teacher_m, teacher_t = loader.load_teacher_model()
student_m, student_t = loader.load_student_model()

# Create wrappers
teacher = TeacherModel(teacher_m, teacher_t)
student = StudentModel(student_m, student_t)

# Compare
comparator = ComparativeInference(teacher, student)

image = Image.open("test.jpg")
question = "Có bao nhiêu người trong ảnh?"

comparison = comparator.compare_single(image, question)
print(f"Teacher: {comparison['teacher_answer']}")
print(f"Student: {comparison['student_answer']}")
```

### Example 3: Batch Inference

```python
from models import VQAModelLoader, TeacherModel, VQAInference

loader = VQAModelLoader()
model, tokenizer = loader.load_teacher_model()
teacher = TeacherModel(model, tokenizer)
inference = VQAInference(teacher)

# Load images and questions
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
questions = [
    "Hãy mô tả bức ảnh này",
    "Có bao nhiêu người?",
    "Màu chủ đạo là gì?",
    "Trong lớp ảnh nào?",
    "Đây là ở đâu?"
]

# Batch inference
answers = inference.predict_batch(images, questions)
for q, a in zip(questions, answers):
    print(f"Q: {q} -> A: {a}")
```

## Configuration

Model loading can be configured via `configs/model_config.yaml`:

```yaml
model_teacher:
  local: "outputs-7B/final_lora"
  huggingface: "MinhQuy24/Qwen-7B-Vi-VQA"
  max_seq_length: 2048
  load_in_4bit: false

model_student:
  name: "Qwen/Qwen2-VL-2B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
```

## Performance Tips

1. **Memory**: Use `load_in_4bit=True` for 8GB+ GPUs
2. **Speed**: Use `torch.inference_mode()` (already built-in)
3. **Batch**: Process multiple samples at once
4. **Caching**: Models cache after first load

## API Reference

### VQAModelLoader

- `load_teacher_model()` to (model, tokenizer)
- `load_student_model()` to (model, tokenizer)
- `load_base_model()` to (model, tokenizer)
- `get_model(name)` to model or None
- `list_loaded_models()` to list
- `unload_model(name)` to None

### VQAInference

- `predict_single(image, question, max_tokens)` to str
- `predict_batch(images, questions, max_tokens)` to List[str]
- `predict_from_file(path, question, max_tokens)` to str
- `predict_from_json(json_path, max_tokens, limit)` to List[Dict]
- `save_predictions(predictions, output_path)` to None

### TeacherModel / StudentModel

- `answer_question(image, question, max_tokens)` to str
- `generate(images, questions, max_tokens, ...)` to List[str]
- `print_model_info()` to None
- `to(device)` to None

## Troubleshooting

**Q: Out of Memory Error**
- Use `load_in_4bit=True` for quantization
- Reduce batch size
- Use smaller model (2B instead of 7B)

**Q: Model Loading Takes Long**
- First load downloads model (~14GB for 7B)
- Subsequent loads are cached
- Use `streaming=True` for very limited memory

**Q: Authentication Error**
- Set `HF_TOKEN` environment variable
- Or pass `use_auth_token=False` if model is public

## Advanced: Custom Model Wrapper

```python
from src.models import VisionLanguageModelWrapper

class CustomVQAModel(VisionLanguageModelWrapper):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(model, tokenizer, "CustomModel", device)
    
    def custom_generate(self, ...):
        # Custom generation logic
        pass
```

# Data Structure - Vi-VQA Dataset

## Dataset Source

The Vi-VQA dataset is stored on HuggingFace Hub:
- **URL**: `MinhQuy24/vlsp2023-vqa-dataset`
- **Type**: Vision Language datasets (Images + Questions + Answers)

## Folder Structure

```
data/
├── raw/                    # Raw data (downloaded from HuggingFace)
│   ├── train/              # Training images
│   ├── dev/                # Validation images
│   └── test/               # Test images
│
├── processed/              # Processed/cleaned data
│   ├── train_processed.json
│   ├── dev_processed.json
│   └── test_processed.json
│
└── cache/                  # Cache for Hugging Face datasets
    └── (auto-generated)
```

## Data Loading

### Option 1: Load from HuggingFace Hub (RECOMMENDED)
```python
from datasets import load_dataset

# Load directly from Hub
dataset = load_dataset("MinhQuy24/vlsp2023-vqa-dataset")

# Access splits
train_dataset = dataset["train"]
dev_dataset = dataset["dev"]
test_dataset = dataset["test"]
```

### Option 2: Load Local JSON Files
```python
import json

with open("raw/train/vlsp2023_train_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
```

## Dataset Format

Expected format from HuggingFace:
```json
{
  "image_id": 12345,
  "question": "Hãy mô tả bức ảnh này",
  "answer": "Một người đi xe đạp trên đường",
  "image_path": "training-images/000000012345.jpg"
}
```

## Data Statistics (Expected)

| Split | Samples | Images |
|-------|---------|--------|
| Train | ~10,000 | ~7,000 |
| Dev   | ~2,000  | ~1,500 |
| Test  | ~2,000  | ~1,500 |

## Important Notes

- Dataset is hosted on HuggingFace - **no need to download manually**
- Images are typically 256×256 or 512×512
- Questions are in Vietnamese
- Answers are short Vietnamese phrases (1-20 words)

## Environment Setup

To download from HuggingFace, you may need to set:
```bash
export HF_TOKEN=your_huggingface_token
```

Or authenticate in Python:
```python
from huggingface_hub import login
login(token="your_token")
```

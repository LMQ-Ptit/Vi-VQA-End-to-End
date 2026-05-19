# Data Loader Usage Example

## Overview
This notebook demonstrates how to load and use the VQA dataset from HuggingFace Hub.

Dataset: `MinhQuy24/vlsp2023-vqa-dataset`

```python
# Import the data loader
import sys
sys.path.insert(0, '../src')

from data import VQADataLoader

# Initialize loader
loader = VQADataLoader(
    repo_id="MinhQuy24/vlsp2023-vqa-dataset",
    cache_dir="./data/cache",
    use_auth_token=True,
)

# Load dataset
dataset = loader.load_dataset()

# Print statistics
loader.print_statistics()

# Print sample
loader.print_sample(split="train", index=0)

# Get individual splits
train_dataset = loader.get_train()
dev_dataset = loader.get_dev()
test_dataset = loader.get_test()

print(f"Train: {len(train_dataset)}")
print(f"Dev: {len(dev_dataset)}")
print(f"Test: {len(test_dataset)}")
```

## Using with Training

```python
from data import prepare_vqa_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Prepare dataset for training
processed_train = prepare_vqa_dataset(
    train_dataset,
    tokenizer,
    max_samples=None,  # Use all samples
    shuffle=True,
)

print(f"Processed train dataset: {len(processed_train)} samples")
print(f"Sample keys: {list(processed_train[0].keys())}")
```

## Features

- ✅ Simple API for loading VQA datasets
- ✅ Automatic HuggingFace Hub integration
- ✅ Dataset statistics and sampling
- ✅ Message formatting for VLM training
- ✅ Support for multiple splits (train, dev, test)

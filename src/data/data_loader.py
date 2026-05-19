"""
Data loading module for Vi-VQA dataset from HuggingFace Hub.
Dataset: MinhQuy24/vlsp2023-vqa-dataset
"""

import os
import json
from typing import Optional, Dict, List, Tuple
from datasets import load_dataset, Dataset


class VQADataLoader:
    """Load VQA dataset from HuggingFace Hub"""
    
    def __init__(
        self,
        repo_id: str = "MinhQuy24/vlsp2023-vqa-dataset",
        cache_dir: str = "./data/cache",
        use_auth_token: bool = True,
        streaming: bool = False,
    ):
        """
        Initialize VQA Data Loader
        
        Args:
            repo_id: HuggingFace repo ID
            cache_dir: Cache directory for downloaded data
            use_auth_token: Use HuggingFace authentication token
            streaming: Stream data without downloading
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
        self.streaming = streaming
        self.dataset = None
        
    def load_dataset(self) -> Dict[str, Dataset]:
        """
        Load dataset from HuggingFace Hub
        
        Returns:
            Dictionary with splits: {'train': Dataset, 'dev': Dataset, 'test': Dataset}
        """
        try:
            print(f"Loading dataset from: {self.repo_id}")
            
            self.dataset = load_dataset(
                self.repo_id,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token,
                streaming=self.streaming,
            )
            
            print(f"✅ Dataset loaded successfully!")
            print(f"Available splits: {list(self.dataset.keys())}")
            
            for split_name in self.dataset.keys():
                print(f"  - {split_name}: {len(self.dataset[split_name])} samples")
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def get_split(self, split_name: str) -> Dataset:
        """
        Get a specific split of the dataset
        
        Args:
            split_name: Name of split ('train', 'dev', 'test')
            
        Returns:
            Dataset for the specified split
        """
        if self.dataset is None:
            self.load_dataset()
        
        if split_name not in self.dataset:
            raise ValueError(
                f"Split '{split_name}' not found. "
                f"Available splits: {list(self.dataset.keys())}"
            )
        
        return self.dataset[split_name]
    
    def get_train(self) -> Dataset:
        """Get training dataset"""
        return self.get_split("train")
    
    def get_dev(self) -> Dataset:
        """Get development/validation dataset"""
        return self.get_split("dev")
    
    def get_test(self) -> Dataset:
        """Get test dataset"""
        return self.get_split("test")
    
    def get_sample(self, split: str = "train", index: int = 0) -> Dict:
        """
        Get a single sample from the dataset
        
        Args:
            split: Dataset split
            index: Sample index
            
        Returns:
            Single sample dictionary
        """
        dataset = self.get_split(split)
        return dataset[index]
    
    def print_sample(self, split: str = "train", index: int = 0) -> None:
        """Print a sample from the dataset for inspection"""
        sample = self.get_sample(split, index)
        
        print(f"\n{'='*60}")
        print(f"Sample from {split} split (index {index})")
        print(f"{'='*60}")
        
        for key, value in sample.items():
            if key == "image":
                print(f"{key}: Image object (shape: {value.size if hasattr(value, 'size') else 'N/A'})")
            else:
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"{key}: {val_str}")
        
        print(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics"""
        if self.dataset is None:
            self.load_dataset()
        
        stats = {}
        for split_name in self.dataset.keys():
            stats[split_name] = len(self.dataset[split_name])
        
        stats['total'] = sum(stats.values())
        return stats
    
    def print_statistics(self) -> None:
        """Print dataset statistics"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        
        for split_name, count in stats.items():
            if split_name != 'total':
                print(f"{split_name:10s}: {count:,} samples")
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':10s}: {stats['total']:,} samples")
        print(f"{'='*60}\n")


def build_vqa_messages(question: str, answer: str) -> List[Dict]:
    """
    Build message format for VLM training
    
    Args:
        question: Question text
        answer: Answer text
        
    Returns:
        List of message dictionaries for chat template
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def prepare_vqa_dataset(
    dataset: Dataset,
    tokenizer,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> Dataset:
    """
    Prepare dataset for training
    
    Args:
        dataset: Raw VQA dataset
        tokenizer: Tokenizer with apply_chat_template method
        max_samples: Maximum samples to use (None = all)
        shuffle: Whether to shuffle dataset
        
    Returns:
        Prepared dataset
    """
    
    def process_sample(sample):
        """Process single sample"""
        messages = build_vqa_messages(sample["question"], sample["answer"])
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "images": [sample["image"]],
            "text": text,
            "question": sample["question"],
            "answer": sample["answer"],
        }
    
    # Process dataset
    processed = dataset.map(process_sample, remove_columns=["question", "answer"])
    
    # Shuffle if requested
    if shuffle:
        processed = processed.shuffle(seed=42)
    
    # Limit samples if specified
    if max_samples is not None:
        processed = processed.select(range(min(max_samples, len(processed))))
    
    return processed


if __name__ == "__main__":
    # Example usage
    loader = VQADataLoader()
    
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
    
    print(f"✅ Loaded datasets successfully!")
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

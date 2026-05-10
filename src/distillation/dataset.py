"""
Distillation Dataset Module
Dataset loader cho knowledge distillation
"""
import torch
from torch.utils.data import DataLoader

class VQADistillDataset(torch.utils.data.Dataset):
    """Dataset class cho distillation training"""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def create_dataloader(dataset, batch_size=2, shuffle=True):
    """Tạo DataLoader cho distillation"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
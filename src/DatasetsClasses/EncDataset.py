import torch
import numpy as np
from torch.utils.data import Dataset

class EncoderDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean) / std

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x, x

    def __len__(self):
        return len(self.data)
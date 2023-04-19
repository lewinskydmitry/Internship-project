import pandas as pd
import torch
from torch.utils.data import Dataset

class TableDatasetPath(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.targets = self.data[self.data.columns[-1]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor(row[:-1], dtype=torch.float32)
        y = torch.tensor(row[-1], dtype=torch.long)
        return x, y
    

class TableDatasetDF(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.targets = self.data[self.data.columns[-1]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor(row[:-1], dtype=torch.float32)
        y = torch.tensor(row[-1], dtype=torch.long)
        return x, y
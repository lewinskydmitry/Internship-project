import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

### MODELS ###
class Baseline_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Baseline_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU(),
            nn.Linear(init_param, int(init_param/2)),
            nn.BatchNorm1d(int(init_param/2)),
            nn.ReLU(),
            nn.Linear(int(init_param/2), int(init_param/4)),
            nn.BatchNorm1d(int(init_param/4)),
            nn.ReLU(),
            nn.Linear(int(init_param/4), int(init_param/8)),
            nn.BatchNorm1d(int(init_param/8)),
            nn.ReLU(),
            nn.Linear(int(init_param/8), int(init_param/16)),
            nn.BatchNorm1d(int(init_param/16)),
            nn.ReLU(),
            nn.Linear(int(init_param/16), int(init_param/32)),
            nn.BatchNorm1d(int(init_param/32)),
            nn.ReLU(),
            nn.Linear(int(init_param/32), int(init_param/64)),
            nn.BatchNorm1d(int(init_param/64)),
            nn.ReLU(),
            nn.Linear(int(init_param/64), int(init_param/128)),
            nn.BatchNorm1d(int(init_param/128)),
            nn.ReLU(),
            nn.Linear(int(init_param/128), 2)
)

    def forward(self, x):
        x = self.classifier(x)
        return x
    
class Simple_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Simple_classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU()
        )

        self.layer2 = nn.Linear(init_param, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

### LOSS FUNCTION ###
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

### DATASETS ###
class TableDatasetPath(Dataset):
    def __init__(self, path):
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]

        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)

        self.features = (self.features - mean) / std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TableDatasetDF(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]

        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0)

        self.features = (self.features - mean) / std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
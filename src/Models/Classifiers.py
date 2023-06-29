import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

### MODELS ###
class BaselineClassifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(BaselineClassifier, self).__init__()

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
            nn.Linear(int(init_param/64), int(init_param/64)),
            nn.BatchNorm1d(int(init_param/64)),
            nn.ReLU(),
            nn.Linear(int(init_param/64), 2)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, num_features):
        super(SimpleClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


### LOSS FUNCTION ###
class LossWrapper:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, y_pred, y_true, *args, **kwargs):
        y_pred = torch.softmax(y_pred, dim=1)
        out = self.loss(y_pred, y_true, *args, **kwargs)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        f1_sc = metrics.f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.cpu().numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:, 1])
        auc_score = metrics.auc(fpr, tpr)

        tp = ((y_pred[:, 1] >= 0.5) & (y_true == 1)).sum()
        fp = ((y_pred[:, 1] >= 0.5) & (y_true == 0)).sum()
        tn = ((y_pred[:, 1] < 0.5) & (y_true == 0)).sum()
        fn = ((y_pred[:, 1] < 0.5) & (y_true == 1)).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        return out, {
            'loss': out.item(),
            'accuracy': accuracy.item(),
            'f1_score': f1_sc,
            'auc_score': auc_score,
            'tpr': tpr,
            'fpr': fpr
        }



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
        

class DWBLoss(nn.Module):
    def __init__(self, labels):
        super(DWBLoss, self).__init__()
        self.class_weights = self.get_class_weights(labels)

    def get_class_weights(self, labels):
        class_counts = torch.bincount(labels)
        max_class_count = torch.max(class_counts)
        class_weights = torch.log(max_class_count / class_counts) + 1
        return class_weights

    def forward(self, logits, targets):

        class_probabilities = F.softmax(logits, dim=1)
        log_class_probabilities = F.log_softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=logits.size(1))

        loss = torch.mean(
            -self.class_weights.pow(1 - class_probabilities) * one_hot_targets * log_class_probabilities
        ) - torch.mean(class_probabilities * (1 - class_probabilities))

        return loss
    

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, target, input):
        log_softmax = torch.log_softmax(input, dim=1)
        loss = -torch.mean(torch.sum(log_softmax * target, dim=1))
        return loss


### DATASETS ###
class ClassifierDataset(Dataset):
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
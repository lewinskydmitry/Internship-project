from sklearn.metrics import f1_score
from sklearn import metrics
import torch

class Loss_class:
    def __init__(self,loss):
        self.loss = loss
        
    def __call__(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        out = self.loss(y_pred, y_true)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        f1_sc = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
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
        return out, {'loss': out.item(), 'accuracy': accuracy.item(),
                   'f1_score': f1_sc,
                   'auc_score': auc_score,
                   'tpr':tpr,
                   'fpr':fpr}


class Loss_class_TH:
    def __init__(self,loss):
        self.loss = loss
        
    def __call__(self, features, y_pred, y_true, x_init, x_restored):
        y_pred = torch.softmax(y_pred, dim=1)
        out = self.loss(features, y_pred, y_true, x_init, x_restored)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        f1_sc = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
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
        return out, {'loss': out.item(), 'accuracy': accuracy.item(),
                   'f1_score': f1_sc,
                   'auc_score': auc_score,
                   'tpr':tpr,
                   'fpr':fpr}
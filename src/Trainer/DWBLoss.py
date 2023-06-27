import torch
import torch.nn as nn
import torch.nn.functional as F


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

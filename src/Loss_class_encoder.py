import torch

class Loss_class_encoder:
    def __init__(self, loss):
        self.loss = loss
        
    def __call__(self, y_pred, y_true):
        out = self.loss(y_pred, y_true)
        
        return out, {'loss': out.item()}
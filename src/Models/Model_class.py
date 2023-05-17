import torch.nn as nn

class Model_class(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
    def __call__(self, x):
        return self.model(x)

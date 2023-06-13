import torch.nn as nn


class JoinedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(JoinedModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.classifier.classifier[0] = nn.Linear(self.encoder[-1].out_features,
                                                self.classifier.classifier[0].out_features)
        # Freeze encoder
        for param in encoder.parameters():
            param.requires_grad = False

        
    def forward(self, x):
        if isinstance(self.encoder, nn.Sequential):
            x = self.encoder(x)
            x = self.classifier(x)
        else:
            mean, logvar = self.encoder.encode(x)
            x = self.classifier(mean)
        return x
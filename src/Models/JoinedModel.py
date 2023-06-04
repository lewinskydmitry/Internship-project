import torch.nn as nn

class JoinedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(JoinedModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

        # Get list of encoder layers
        if isinstance(encoder, nn.Sequential):
            # Fix dimensionality of classifier input layer
            classifier.layer1[0] = nn.Linear(self.encoder[-1].out_features, classifier.layer1[0].out_features)
            # Freeze encoder
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            # Fix dimensionality of classifier input layer
            classifier.layer1[0] = nn.Linear(self.encoder.latent_size, classifier.layer1[0].out_features)
            
            # Freeze encoder
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        if isinstance(self.encoder, nn.Sequential):
            x = self.encoder(x)
            x = self.classifier(x)
        else:
            mean, logvar = self.encoder.encode(x)
            x = self.classifier(mean)
        return x
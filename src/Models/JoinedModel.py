import torch.nn as nn

class JoinedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(JoinedModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

        # Get list of encoder layers
        if isinstance(encoder, nn.Sequential):
            encoder_layers = list(encoder.children())
        else:
            encoder_layers = [encoder]
        
        # Fix dimensionality of classifier input layer
        classifier.layer1[0] = nn.Linear(encoder_layers[-1].out_features, classifier.layer1[0].out_features)
        
        # Freeze encoder
        for param in encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class JoinedModel_VAE(nn.Module):
    def __init__(self, vae_model, classifier):
        super(JoinedModel_VAE, self).__init__()
        self.vae = vae_model
        self.classifier = classifier
        
        # Fix dimensionality of classifier input layer
        classifier.layer1[0] = nn.Linear(self.vae.latent_size, classifier.layer1[0].out_features)
        
        # Freeze encoder
        for param in vae_model.decoder.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        mean, logvar = self.vae.encode(x)
        x = self.classifier(mean)
        return x
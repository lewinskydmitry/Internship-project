import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import copy

def create_mirror_layers(layers):
    mirror_modules = []
    for module in reversed(layers):
        if isinstance(module, nn.Linear):
            mirror_modules.append(nn.Linear(module.out_features, module.in_features))
        else:
            mirror_modules.append(module)
    return nn.Sequential(*mirror_modules)

### MODELS ###
class Autoencoder(nn.Module):
    def __init__(self, encoder, latent_space = None):
        super(Autoencoder, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        if latent_space != None:
            self.encoder[-1] = nn.Linear(self.encoder[-1].in_features, latent_space)
        self.decoder = create_mirror_layers(self.encoder)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self,encoder, latent_size):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = copy.deepcopy(encoder)
        self.latent_size = latent_size
        self.encoder[-1] = nn.Linear(self.encoder[-1].in_features, self.encoder[-1].out_features*2)
        if self.latent_size != None:
            self.encoder[-1] = nn.Linear(self.encoder[-1].in_features, self.latent_size*2)
        self.decoder = create_mirror_layers(encoder)
        self.decoder[0] = nn.Linear(self.latent_size, self.decoder[0].out_features)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, logvar = torch.split(encoded, self.latent_size, dim=1)  # Split the encoded tensor into mean and logvar
        return mean, logvar

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

### LOSS FUNCTION ###
class Encoder_loss:
    def __init__(self, loss):
        self.loss = loss
        
    def __call__(self, y_pred, y_true):
        out = self.loss(y_pred, y_true)
        
        return out, {'loss': out.item()}
    
class vae_loss:  
    def __init__(self, loss):
        self.loss = loss
          
    def __call__(self, recon_x, mean, logvar, x):
        out = self.loss(recon_x, mean, logvar, x)
        
        return out, {'loss': out.item()}

def vae_loss_function(recon_x, mean, logvar, x):
    # Reconstruction losss

    reconstruction_loss = F.mse_loss(x,recon_x)

    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss + kl_divergence_loss*10e-4

    return total_loss

### DATASETS ###
class EncoderDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean) / std

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x, x

    def __len__(self):
        return len(self.data)

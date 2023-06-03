import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

### MODELS ###
class Autoencoder(nn.Module):
    def __init__(self,encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self,encoder, decoder):
        super(VAE, self).__init__()

        # Encoder layers
        self.encoder = encoder
        self.latent_size = encoder[-1].out_features
        self.encoder[-1] = nn.Linear(encoder[-1].in_features,encoder[-1].out_features*2)
        # Decoder layers
        self.decoder = decoder

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

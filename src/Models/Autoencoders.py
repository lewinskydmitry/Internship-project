import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

### MODELS ###
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_representation):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), latent_representation)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_representation, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), input_size)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mean = self.fc_mean(h1)
        logvar = self.fc_logvar(h1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

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
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss + kl_divergence_loss

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

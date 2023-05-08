import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_representation):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/4), latent_representation)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_representation, int(hidden_size/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/4), int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), input_size)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
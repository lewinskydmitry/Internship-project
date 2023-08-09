import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('src'), '..')))

from src.models.autoencoders import create_mirror_layers

import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch
import copy
import warnings


class GevActivation(nn.Module):
    def __init__(self):
        super(GevActivation, self).__init__()

    def forward(self, x):
        return torch.exp(-torch.exp(-x))

class GevNN(nn.Module):
    def __init__(self, encoder, weighted_model, main_classifier, latent_space = None):
        super(GevNN, self).__init__()

        # Copy nn.Sequential for model parts
        self.weighted_model = copy.deepcopy(weighted_model)
        self.main_classifier = copy.deepcopy(main_classifier)

        # Create decoder and change encoder if it's necessary
        self.encoder = copy.deepcopy(encoder)
        if latent_space != None:
            self.encoder[-1] = nn.Linear(self.encoder[-1].in_features, latent_space)
        self.decoder = create_mirror_layers(self.encoder)

        # Check that dimentions are correct
        ############################################################################
        if self.weighted_model[-1].out_features != self.weighted_model[0].in_features:
            raise ValueError(f"Input and output dimentions of weighted NN should be equal. Now In-{self.weighted_model[-1].out_features}\
                              and out-{self.weighted_model[0].in_features}")

        if self.encoder[-1].out_features != self.decoder[0].in_features:
            raise ValueError(f"Dimentions of encoder-decoder don't match")
        
        concat_dim = self.encoder[-1].out_features + weighted_model[0].in_features + 2
        
        if self.main_classifier[0].in_features != concat_dim:
            warnings.warn("Dimentions for input of main_classifier is changed because of mismatch")
            self.main_classifier[0] = nn.Linear(concat_dim, main_classifier[0].out_features)
        ############################################################################
        self.gev_activation = GevActivation()


    def euclidean_distance(self, A, B):
        euclidean_dist = torch.norm(A - B, dim=1)
        euclidean_dist = euclidean_dist.reshape(-1,1)
        return euclidean_dist
    

    def cosine_distance(self, A, B):
        cosine_sim = cosine_similarity(A, B)
        cosine_dist = 1 - cosine_sim
        cosine_dist = cosine_dist.reshape(-1,1)
        return cosine_dist


    def forward(self, x):
        importances = self.weighted_model(x)
        selected_input = torch.mul(x, torch.softmax(importances,dim=1))
        latent_space = self.encoder(x)
        restored_x = self.decoder(latent_space)
        cos_dist = self.cosine_distance(x, restored_x)
        euclid_dist = self.euclidean_distance(x, restored_x)
        final_input = torch.concatenate([selected_input, latent_space, cos_dist, euclid_dist], dim=1)
        output = self.main_classifier(final_input)
        output = self.gev_activation(output)
        return output, x, restored_x


class GevLoss(nn.Module):
    def __init__(self, loss_classif):
        super(GevLoss, self).__init__()
        self.loss_classif = loss_classif
        self.rest_loss = nn.MSELoss()

    def forward(self, y_pred, y_true, x, restored_x):
        classif_loss = self.loss_classif(y_pred, y_true.float())
        ae_loss = self.rest_loss(x, restored_x)  
        out = torch.mean(classif_loss + ae_loss)

        return out

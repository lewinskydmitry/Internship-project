import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from src.models.autoencoders import create_mirror_layers
import copy

class TwoHeadedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(TwoHeadedModel, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = create_mirror_layers(self.encoder)
        self.classifier = copy.deepcopy(classifier)
        self.classifier[0] = nn.Linear(self.encoder[-1].out_features, self.classifier[0].out_features)

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class TwoHeadedLoss(nn.Module):
    def __init__(self, num_clusters, classification_loss_fn, decoder_loss):
        super(TwoHeadedLoss, self).__init__()
        self.num_clusters = num_clusters
        self.classification_loss_fn = classification_loss_fn
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.decoder_loss = decoder_loss

    def forward(self, features, classification_logits, labels, x_init, x_restored):
        # Update cluster centers using k-means
        self.kmeans.fit(features.detach().cpu().numpy())
        cluster_centers = torch.from_numpy(self.kmeans.cluster_centers_).to(features.device)
        
        # Compute cluster loss
        distances = torch.cdist(features, cluster_centers)
        softmax_weights = nn.functional.softmax(-distances, dim=1)
        cluster_loss = torch.mean(
            softmax_weights[:, 0] * distances[:, 0] + (1 - softmax_weights[:, 0]) * distances[:, 1]
        )
        
        # Compute classification loss
        classification_loss = self.classification_loss_fn(classification_logits, labels)
        decoder_loss = self.decoder_loss(x_init, x_restored)
        
        # Combine the two losses
        total_loss = cluster_loss + classification_loss + decoder_loss
        
        return total_loss
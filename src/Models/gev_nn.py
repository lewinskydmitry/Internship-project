import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn import metrics
import torch
import copy

def create_mirror_layers(layers):
    mirror_modules = []
    for module in reversed(layers):
        if isinstance(module, nn.Linear):
            mirror_modules.append(nn.Linear(module.out_features, module.in_features))
        else:
            mirror_modules.append(module)
    return nn.Sequential(*mirror_modules)


class GevNN(nn.Module):
    def __init__(self, encoder, weighted_model, main_classifier, latent_space = None):
        super(GevNN, self).__init__()
        self.weighted_model = copy.deepcopy(weighted_model)
        self.main_classifier = copy.deepcopy(main_classifier)

        # Create decoder and change encoder if it's necessary
        self.encoder = copy.deepcopy(encoder)
        if latent_space != None:
            self.encoder[-1] = nn.Linear(self.encoder[-1].in_features, latent_space)
        self.decoder = create_mirror_layers(self.encoder)

        # match dimentions of initial models
        if self.weighted_model[-1].out_features != self.weighted_model[0].in_features:
            raise ValueError(f"Input and output dimentions of weighted NN should be equal. Now In-{self.weighted_model[-1].out_features}\
                              and out-{self.weighted_model[0].in_features}")
        
        if self.encoder[-1].out_features != self.decoder[0].in_features:
            raise ValueError(f"Dimentions of encoder-decoder don't match")
        
    
    def euclidean_distance(self, A, B):
        euclidean_dist = torch.norm(A - B, dim=1)
        return euclidean_dist
    

    def cosine_distance(self, A, B):
        cosine_sim = cosine_similarity(A, B)
        cosine_dist = 1 - cosine_sim
        return cosine_dist


    def forward(self, x):
        importances = self.weighted_model(x)
        selected_input = torch.mul(x, torch.softmax(importances,dim=1))
        latent_space = self.encoder(x)
        restored_x = self.decoder(latent_space)
        cos_dist = self.cosine_distance(x, restored_x)
        euclid_dist = self.euclidean_distance(x, restored_x)
        final_input = torch.concatenate([selected_input, latent_space, cos_dist, euclid_dist])
        output = self.main_classifier(final_input)
        return output, restored_x
    

class GevLoss:  
    def __init__(self, loss_classif):
        self.loss_classif = loss_classif
          
    def __call__(self,  y_pred, y_true, x, restored_x):
        classif_loss = self.loss_classif( y_pred, y_true)
        ae_loss = nn.MSELoss(x, restored_x)
        out = classif_loss + ae_loss

        return out, {'loss': out.item()}
    

class GevLossWrapper:
    def __init__(self,loss):
        self.loss = loss
        
    def __call__(self, y_pred, y_true, x, restored_x):
        y_pred = torch.softmax(y_pred, dim=1)
        out = self.loss(y_pred, y_true, x, restored_x)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        f1_sc = metrics.f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.cpu().numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:, 1])
        auc_score = metrics.auc(fpr, tpr)
        
        tp = ((y_pred[:, 1] >= 0.5) & (y_true == 1)).sum()
        fp = ((y_pred[:, 1] >= 0.5) & (y_true == 0)).sum()
        tn = ((y_pred[:, 1] < 0.5) & (y_true == 0)).sum()
        fn = ((y_pred[:, 1] < 0.5) & (y_true == 1)).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return out, {'loss': out.item(), 'accuracy': accuracy.item(),
                   'f1_score': f1_sc, 'auc_score': auc_score,
                   'tpr':tpr, 'fpr':fpr}
    

Questions:
- How to name losses classes?
- Is Gev correct?

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from catboost import CatBoostClassifier
from tqdm import tqdm
from catboost import Pool
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Subset
from src.Focal_loss import *
from functools import partial
from src.trainer import Trainer
import torch.optim.lr_scheduler as lr
import torch
from torch.utils.data import DataLoader
from src.DataLoader import TableDatasetPath, TableDatasetDF
import wandb


def upsampling(p=1., *data):
  if len(data) == 1:
    # Count the number of samples in each class
    df = data[0]
    class_counts = df['Machine failure'].value_counts()

    # Determine the minority and majority class
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Upsample the minority class with replacement
    minority_df = df[df['Machine failure'] == minority_class]
    upsampled_df = pd.concat(
        [df] + [minority_df.sample(n=int((class_counts[majority_class] - class_counts[minority_class])*p),
                                  replace=True)], axis=0)

    # Shuffle the upsampled data
    upsampled_df = upsampled_df.sample(frac=1).reset_index(drop = True)
    return upsampled_df
  else:
    df = pd.concat([data[0], data[1]], axis=1)
    # Count the number of samples in each class
    class_counts = df['Machine failure'].value_counts()

    # Determine the minority and majority class
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Upsample the minority class with replacement
    minority_df = df[df['Machine failure'] == minority_class]
    upsampled_df = pd.concat(
        [df] + [minority_df.sample(n=int((class_counts[majority_class] - class_counts[minority_class])*p),
                                  replace=True)], axis=0)

    # Shuffle the upsampled data
    upsampled_df = upsampled_df.sample(frac=1).reset_index(drop = True)
    return upsampled_df.drop(columns = ['Machine failure']), upsampled_df['Machine failure']
  


def check_result(model,X_test,y_test):
  metrics_dict = {}
  fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
  auc_score = metrics.auc(fpr, tpr)
  f1_sc = f1_score(y_test, model.predict(X_test), average='macro')
  metrics_dict['auc_score'] = auc_score
  metrics_dict['f1_score'] = f1_sc
  return metrics_dict


def search_num_features(df, feature_importance, upsamp_func = False, step = 5):
  best_score = 0
  best_num_features = 0
  for num_col in tqdm(range(1, len(feature_importance), step)):
    features = list(feature_importance.iloc[:num_col,:]['feature_names'])
    data = df[features + ['Machine failure']]
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = ['Machine failure']),
                                                        data['Machine failure'],
                                                        test_size=0.33,
                                                        random_state=42,
                                                        stratify = df['Machine failure'])
    if upsamp_func == True:
      X_train, y_train = upsampling(X_train, y_train)

    train_pool = Pool(data=X_train, label=y_train)
    CatBoost = CatBoostClassifier(verbose=False)
    CatBoost.fit(train_pool)
    metrics_dict = check_result(CatBoost, X_test, y_test)
    print(f'F1_score - {metrics_dict["f1_score"]}, num_features - {best_num_features}, AUC_score = {metrics_dict["auc_score"]}')
    if metrics_dict['f1_score'] > best_score:
      best_score = metrics_dict['f1_score']
      best_num_features = num_col
  print(f'Best F1_score - {best_score}, num_features - {best_num_features}')

###########################################################################################

class Model_class(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
    def __call__(self, x):
        return self.model(x)


class Loss_class:
    def __init__(self,loss):
        self.loss = loss
        
    def __call__(self, y_pred, y_true):
      out = self.loss(y_pred, y_true)
      y_pred = torch.softmax(y_pred, dim=1)
      accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
      f1_sc = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
      y_pred = y_pred.detach().cpu().numpy()
      y_true = y_true.cpu().numpy()
      fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:, 1])
      auc_score = metrics.auc(fpr, tpr)
      return out, {'loss': out.item(), 'accuracy': accuracy.item(), 'f1_score': f1_sc, 'auc_score': auc_score}



def balance_val_split(dataset, train_size=0.7):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        shuffle=True,
        train_size=train_size,
        stratify=targets
    )

    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset
######################################


def make_data(num_features, batch_size):
    
    df = pd.read_csv('../data/data_fe.csv') # HARD CODE FOR PATHES!

    feature_importance = pd.read_csv('../data/feature_imporstance_WS.csv') # HARD CODE FOR PATHES!

    features = list(feature_importance.iloc[:num_features]['feature_names'])
    df = df[features + ['Machine failure']]

    X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Machine failure']),
                                                    df['Machine failure'],
                                                    shuffle=True,
                                                    stratify=df['Machine failure'], random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    train_dataset = TableDatasetDF(df_train)
    val_dataset = TableDatasetDF(df_test)

    train_dl = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=True)
        
    return train_dl, val_dl


def create_nn_with_data(num_features,
                        batch_size,
                        init_param,
                        num_layers):
    
    train_dl, val_dl = make_data(num_features, batch_size)

    decay = 2
    model = nn.Sequential()
    model.add_module('Linear_0', nn.Linear(train_dl.dataset.data.shape[1]-1, init_param))
    model.add_module('BatchNorm1d_0', nn.BatchNorm1d(init_param))
    model.add_module('Dropout_0', nn.Dropout(0.1))
    model.add_module('Relu_0', nn.ReLU())
    
    for i in range(num_layers-2):
        model.add_module(f'Linear_{i+1}', nn.Linear(int(init_param/decay**i), int(init_param/decay**(i+1))))
        model.add_module(f'BatchNorm1d_{i+1}', nn.BatchNorm1d(int(init_param/decay**(i+1))))
        model.add_module(f'Dropout_{i+1}', nn.Dropout(0.1))
        model.add_module(f'Relu_{i+1}', nn.ReLU())

    model.add_module(f'Linear_{num_layers}', nn.Linear(int(init_param/decay**(num_layers-2)), 2))

    return model, train_dl, val_dl


def make_experiments(model,
                     train_dl,
                     val_dl, batch_size, device = 'cuda'):
  
  loss = Loss_class(FocalLoss(gamma=3))
  model_factory = partial(Model_class)
  optimizer_factory = partial(torch.optim.AdamW)
  scheduler_factory = partial(lr.ExponentialLR)

  model_params = dict(model=model,
                      device=device)
  optimizer_params = dict(weight_decay=1e-3, lr=1e-2)
  scheduler_params = dict(gamma=0.95)

  learning_params = dict(batch_size=batch_size, num_epoch=20)

  wandb_init_params = dict(
      name=f'fe_{model[0].weight.shape[0]}_{batch_size}_{model[0].weight.shape[1]}',
      project="Internship_project",
      dir = '../logs/'
  )
  trainer = Trainer(train_dl,
                  val_dl,
                  loss,
                  model_factory=model_factory,
                  optimizer_factory=optimizer_factory,
                  scheduler_factory=scheduler_factory,
                  model_params=model_params,
                  optimizer_params=optimizer_params,
                  scheduler_params=scheduler_params,
                  log=True,
                  wandb_init_params=wandb_init_params,
                  model_dir='../logs/nn_models/'
                  )
  trainer.train_model(learning_params)
  wandb.finish()
   
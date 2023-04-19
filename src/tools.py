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

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler


def upsampling(X_train, y_train):
  df = pd.concat([X_train, y_train],axis = 1)
  # Count the number of samples in each class
  class_counts = df['Machine failure'].value_counts()

  # Determine the minority and majority class
  minority_class = class_counts.idxmin()
  majority_class = class_counts.idxmax()

  # Upsample the minority class with replacement
  minority_df = df[df['Machine failure'] == minority_class]
  upsampled_df = pd.concat(
      [df] + [minority_df.sample(n=class_counts[majority_class] - class_counts[minority_class],
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
  f1_sc = 0
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
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        f1_sc = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(),average='macro')
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred.argmax(dim=1).cpu())
        auc_score = metrics.auc(fpr, tpr)
        return out, {'loss': out.item(), 'accuracy': accuracy.item(), 'f1_score': f1_sc, 'auc_score':auc_score}


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
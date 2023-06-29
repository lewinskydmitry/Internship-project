import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from catboost import CatBoostClassifier
from tqdm import tqdm
from catboost import Pool
from sklearn.model_selection import train_test_split
from src.sampling_methods.sampler import DataSampler


def check_result(model,X_test,y_test):
  metrics_dict = {}
  fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
  auc_score = metrics.auc(fpr, tpr)
  f1_sc = f1_score(y_test, model.predict(X_test), average='macro')
  metrics_dict['auc_score'] = auc_score
  metrics_dict['f1_score'] = f1_sc
  return metrics_dict


def search_num_features(df, feature_importance, upsamp_func = 'ROS', p = 0.3, step = 5):
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
      X_train, y_train = DataSampler.ROS(X_train, y_train,)

    train_pool = Pool(data=X_train, label=y_train)
    CatBoost = CatBoostClassifier(verbose=False,random_seed=42)
    CatBoost.fit(train_pool)
    metrics_dict = check_result(CatBoost, X_test, y_test)
    print(f'F1_score - {metrics_dict["f1_score"]}, num_features - {best_num_features}, AUC_score = {metrics_dict["auc_score"]}')
    if metrics_dict['f1_score'] > best_score:
      best_score = metrics_dict['f1_score']
      best_num_features = num_col
  print(f'Best F1_score - {best_score}, num_features - {best_num_features}')


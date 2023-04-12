import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from catboost import CatBoostClassifier
from tqdm import tqdm
from catboost import Pool
from sklearn.model_selection import train_test_split


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
  fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
  result = metrics.auc(fpr, tpr)
  acc = accuracy_score(y_test, model.predict(X_test))
  print(f'Accuracy = {acc}, AUC = {result}')
  return result


def search_num_features(df, feature_importance, upsamp_func = False, step = 3):
  best_score = 0
  best_num_features = 0
  acc = 0
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
    metric = check_result(CatBoost, X_test, y_test)
    acc_current = accuracy_score(y_test, CatBoost.predict(X_test))
    if metric > best_score:
      best_score = metric
      best_num_features = num_col
    if acc < acc_current:
      acc = acc_current
  print(f'Best AUC - {best_score}, num_features - {best_num_features}, acc = {acc}')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim.lr_scheduler as lr
import torch

from src.DataLoader import TableDatasetDF
from src.Focal_loss import FocalLoss
from src.Loss_class import Loss_class
from src.Model_class import Model_class
from src.trainer import Trainer

from functools import partial

import wandb


class nn_creater:
    def __init__(self, path_to_data,
                 path_to_featureImportance,
                 num_features_list, init_param_list, num_layers_list, batch_sizes_list):
        self.path_to_data = path_to_data
        self.path_to_featureImportance = path_to_featureImportance
        self.num_features_list = num_features_list
        self.init_param_list = init_param_list
        self.num_layers_list = num_layers_list
        self.batch_sizes_list = batch_sizes_list


    def make_data(self, num_features, batch_size):
    
        df = pd.read_csv(self.path_to_data) 

        feature_importance = pd.read_csv(self.path_to_featureImportance)

        features = list(feature_importance.iloc[:num_features]['feature_names'])
        df = df[features + ['Machine failure']]

        X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Machine failure']),
                                                        df['Machine failure'],
                                                        shuffle=True,
                                                        stratify=df['Machine failure'], random_state=42)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        
        X_train = pd.DataFrame(X_train).reset_index(drop = True)
        X_test = pd.DataFrame(X_test).reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis = 1)

        df_train.columns = features + ['Machine failure']
        df_test.columns = features + ['Machine failure']

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


    def create_nn_with_data(self, num_features,
                            batch_size,
                            init_param,
                            num_layers):
        
        train_dl, val_dl = self.make_data(num_features, batch_size)

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

        model.add_module(f'Linear_{num_layers-1}', nn.Linear(int(init_param/decay**(num_layers-2)), 2))

        return model, train_dl, val_dl
    
    
    def make_experiments(self, model, 
                         train_dl, 
                         val_dl, batch_size, device = 'cuda'):
  
        loss = Loss_class(FocalLoss(gamma=2))
        model_factory = partial(Model_class)
        optimizer_factory = partial(torch.optim.AdamW)
        scheduler_factory = partial(lr.ExponentialLR)

        model_params = dict(model=model,
                            device=device)
        optimizer_params = dict(weight_decay=1e-3, lr=1e-2)
        scheduler_params = dict(gamma=0.95)

        learning_params = dict(batch_size=batch_size, num_epoch=40)

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

    def start_experiment(self):
        for batch_size in self.batch_sizes_list:
            for num_features in self.num_features_list:
                for init_param in self.init_param_list:
                    for num_layers in self.num_layers_list:
                        if np.log2(init_param) < num_layers:
                            print(f"it's impossible to construct NN with parameters = {init_param} and layers = {num_layers}")
                        else:
                            model, train_dl, val_dl = self.create_nn_with_data(num_features, batch_size, init_param, num_layers)
                            self.make_experiments(model, train_dl, val_dl, batch_size)
                            
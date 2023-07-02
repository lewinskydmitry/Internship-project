import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('src'), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr
from src.models.classifiers import *
from src.trainer.trainer import TrainerClassifier, Model_class
from src.sampling_methods.sampler import DataSampler

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
generator = torch.Generator()
generator.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Experimenter():

    def __init__(self, df, parameters = {
        'sampling_methods':[],
        'loss_functions':[nn.CrossEntropyLoss()],
        'p':[0.1,0.2]}, batch_size = 1024):
        self.df = df
        self.num_features = df.shape[1]-1
        self.parameters = parameters
        self.batch_size = batch_size
        self.init_parameters = 512


    def prepare_data(self, sampling = None, p = 0.):
        X_train,X_test,y_train,y_test = train_test_split(self.df.drop(columns=['Machine failure']),
                                                 self.df['Machine failure'],
                                                 shuffle=True,
                                                 stratify=self.df['Machine failure'], random_state=42,
                                                 train_size=0.7)
        if sampling != None:
            sampler = DataSampler()
            if sampling == 'ROS':
                df_train = sampler.ROS(X_train, y_train, p)
            elif sampling == 'RUS':
                df_train = sampler.RUS(X_train, y_train, p)
            elif sampling == 'SMOTE':
                df_train = sampler.SMOTE(X_train, y_train, p)
            elif sampling == 'OSS':
                df_train = sampler.OSS(X_train, y_train)
        else:
            df_train = pd.concat([X_train, y_train], axis = 1)

        df_test = pd.concat([X_test, y_test], axis = 1)
        train_dataset = ClassifierDataset(df_train)
        val_dataset = ClassifierDataset(df_test)

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator
        )

        return train_dl, val_dl
    

    def setup(self, model, train_dl, val_dl, loss_func, sampling = None, p = 0.):
        loss = LossWrapper(loss_func)
        model_factory = partial(Model_class)
        optimizer_factory = partial(torch.optim.AdamW)
        scheduler_factory = partial(lr.ExponentialLR)

        model_params = dict(model=model,
                            device=device)

        optimizer_params = dict(weight_decay=1e-3, lr=1e-2)
        scheduler_params = dict(gamma=0.90)

        wandb_init_params = dict(
            name=f'EM_{loss.__dict__["loss"]}_{sampling}-{p}',
            project="Internship_project",
            dir = '../logs/',
            
        )


        additional_params = dict(loss = loss_func,
                                p = p,
                                sampling = sampling,
                                batch_size = self.batch_size,
                                init_parameters = self.init_parameters)

        trainer = TrainerClassifier(train_dl,
                  val_dl,
                  loss,
                  model_factory=model_factory,
                  optimizer_factory=optimizer_factory,
                  scheduler_factory=scheduler_factory,
                  model_params=model_params,
                  optimizer_params=optimizer_params,
                  scheduler_params=scheduler_params,
                  additional_params = additional_params,
                  log=True,
                  wandb_init_params=wandb_init_params,
                  model_dir='../logs/nn_models/classifier/',
                  saving_model=False
                  )
        
        return trainer


    def perform_experiments(self):
        for loss in self.parameters['loss_functions']:
            for sampling in self.parameters['sampling_methods']:

                if sampling not in [None,'OSS'] :
                    for p in self.parameters['p']:
                        train_dl, val_dl = self.prepare_data(sampling, p)

                        model = BaselineClassifier(self.num_features, self.init_parameters)

                        learning_params = dict(batch_size=self.batch_size, num_epoch=40)
                        trainer = self.setup(model, train_dl, val_dl, loss, sampling, p)
                        trainer.train_model(learning_params)
                        wandb.finish()
                elif sampling == 'OSS':
                    train_dl, val_dl = self.prepare_data(sampling, p)

                    model = BaselineClassifier(self.num_features, self.init_parameters)

                    learning_params = dict(batch_size=self.batch_size, num_epoch=40)
                    trainer = self.setup(model, train_dl, val_dl, loss, sampling, p)
                    trainer.train_model(learning_params)
                    wandb.finish()
                else:
                    train_dl, val_dl = self.prepare_data()

                    model = BaselineClassifier(self.num_features, self.init_parameters)

                    learning_params = dict(batch_size=self.batch_size, num_epoch=40)
                    trainer = self.setup(model, train_dl, val_dl, loss)
                    trainer.train_model(learning_params)
                    wandb.finish()
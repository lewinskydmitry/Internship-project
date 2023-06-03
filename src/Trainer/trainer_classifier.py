import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

import numpy as np
np.random.seed(42)

import time
import wandb

from tqdm import tqdm
from collections import defaultdict


class Trainer_classifier:
    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 loss,
                 model_factory,
                 optimizer_factory,
                 scheduler_factory,
                 model_params,
                 optimizer_params,
                 scheduler_params,
                 log=False,
                 wandb_init_params=None,
                 desc=None,
                 model_dir=None,
                 saving_model = False):
        """
            Class for entire model training and validation process.
            It implements an initialization of the model, optimizer and scheduler.
            It connents with wandb project and logs results.
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss = loss

        # initializing
        self.model = model_factory(**model_params).to(model_params['device'])
        self.optimizer = optimizer_factory(self.model.parameters(), **optimizer_params)
        self.scheduler = scheduler_factory(self.optimizer, **scheduler_params)

        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        # creating a new experiment and log hyperparams
        self.log = log
        self.desc = desc if desc is not None else ''
        self.model_dir = model_dir if model_dir is not None else './'

        self.wandb_init_params = wandb_init_params
        if self.wandb_init_params is None:
            self.wandb_init_params = {
                'name': 'empty_name'
            }
        self.run = None
        #if self.log:
        self.run = self.log_hyperparams(self.wandb_init_params,
                                            **self.model_params, 
                                            **self.optimizer_params, 
                                            **self.scheduler_params)
        # initializing metrics
        self.metrics = None
        self.init_metrics()
        self.saving_model = saving_model


    @staticmethod
    def log_hyperparams(wandb_init_params, **hyperparams):
        run = wandb.init(**wandb_init_params, config={})
        wandb.config.update(hyperparams)
        return run


    @staticmethod
    def log_metrics(metrics):
        wandb.log(metrics)


    def init_metrics(self):
        self.metrics = defaultdict(list)


    def update_metrics(self, **to_update):
        #if self.log:
        self.log_metrics(to_update)
        for name, value in to_update.items():
            self.metrics[name].append(value)


    def train_model(self, learning_params):
        num_epoch = learning_params['num_epoch']
        for epoch in range(num_epoch):
            start_time = time.time()

            train_to_update = self.train_epoch()
            val_to_update = self.validate_epoch(self.test_dataloader, log_folder='test')
            if self.log:
                name = self.wandb_init_params['name']
                model_artifact = wandb.Artifact(name, type="model", description=f"{self.desc}")
                
                log_path = self.model_dir + name + "_state_dict" + ".pth"
                torch.save(self.model.state_dict(), log_path)
                model_artifact.add_file(log_path)
                wandb.save(log_path, base_path = self.wandb_init_params['dir'])

                self.run.log_artifact(model_artifact)
            self.update_metrics(epoch=epoch,
                                **train_to_update, 
                                **val_to_update, 
                                lr=self.scheduler.get_last_lr()[0]
                               )
            
            print("Epoch: {} of {}, {:.3f} min".format(epoch + 1, num_epoch, (time.time() - start_time) / 60))
        if self.saving_model == True:
            torch.save(self.model.model, self.model_dir + self.wandb_init_params['name'] + ".pth")


    def train_epoch(self, dataloader=None):
        dataloader = dataloader if dataloader is not None else self.train_dataloader
        model = self.model
        device = model.device
        opt = self.optimizer
        scheduler = self.scheduler
        
        metrics = defaultdict(list)
        model.train(True)
        for X_batch, y_batch in tqdm(dataloader, desc='I\'m studying hard now🧐, don\'t disturb!'):
            opt.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss, cur_metrics = self.loss(logits, y_batch)
            loss.backward()
            opt.step()

            for name, value in cur_metrics.items():
                metrics[name].append(value)
        scheduler.step()
        to_update = {'train_' + name: np.mean(values) for name, values in metrics.items()}
        return to_update


    def validate_epoch(self, dataloader=None, log_folder='test'):
        dataloader = dataloader if dataloader is not None else self.test_dataloader
        model = self.model
        device = model.device

        metrics = defaultdict(list)
 
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in tqdm(dataloader, desc='Let\'s see how good I am...'):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss, cur_metrics = self.loss(logits, y_batch)

                for name, value in cur_metrics.items():
                    metrics[name].append(value)
        
        to_update = {log_folder + '_' + name: np.mean(values) for name, values in metrics.items()}
        return to_update

import json
import time
from pathlib import Path
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import librosa
import soundfile as sf
from pytorch_mlp_framework.storage_utils import save_statistics

from IPython.display import Audio, display


class Experiment(nn.Module):
    """
    PyTorch experiment framework that can train models and resume older experiments.
    """
    
    def __init__(self, model, experiment_name, num_epochs, train_data, val_data, continue_from_epoch=-1, lr_decay=0.98,
                 use_gpu=True, save_every_epoch=True):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param model: A pytorch nn.Module which implements a network architecture.
        # Possibly separate this into encoder/decoder/discriminator if more steps are needed
        :param experiment_name: str to name the experiment, or Path pointing to an existing experiment_folder
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        
        """
        super(Experiment, self).__init__()
        
        self.model = model
        
        self.lossgrad = []
        self.lossgrad_epoch = []
        
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)
        
        # self.model.reset_parameters()  # re-initialize network parameters
        
        self.train_data = train_data
        self.val_data = val_data
        
        self.optimizer = optim.Adam(self.parameters())
        # Use this for more complex models later
        # self.model_optimizer = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.lr)
        # self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        
        # If experiment_name is already a Path, nothing happens, otherwise later a folder will be made under the
        # relative filename experiment_name
        self.experiment_folder = Path(experiment_name)
        self.experiment_logs = self.experiment_folder / "log"
        self.experiment_checkpoints = self.experiment_folder / "check"
        
        if not self.experiment_folder.is_dir():
            self.experiment_folder.mkdir(parents=True)
            self.experiment_logs.mkdir()
            self.experiment_checkpoints.mkdir()
        
        self.save_every_epoch = save_every_epoch
        self.num_epochs = num_epochs
        
        # -2 means we continue from the latest model
        if continue_from_epoch == -2:
            self.load_model(model_save_name="latest")
        
        elif continue_from_epoch > -1:
            self.load_model(model_save_name="{}".format(continue_from_epoch))
        else:
            self.state = dict()
            self.best_val_score = float('inf')
            self.start_epoch = self.current_epoch = 0
    
    def train_batch(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        # Set system into training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        self.train()
        x, y = x.float().to(device=self.device), y.long().to(device=self.device)  # send data to device as torch tensors
        
        out = self.model.forward(x)
        loss = F.cross_entropy(input=out, target=y)
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        self.lr_scheduler.step(epoch=self.current_epoch)
        
        # Compute argmax of predictions to get accuracy
        _, predicted = torch.max(out.data, 1)
        accuracy = predicted.eq(y.data).cpu().numpy().mean().item()
        return loss.cpu().data.numpy(), accuracy
    
    def eval_batch(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        # Set system to evaluation mode
        self.eval()
        x, y = x.float().to(device=self.device), y.long().to(device=self.device)  # send data to device as torch tensors
        out = self.model.forward(x)
        
        loss = F.cross_entropy(input=out, target=y)  # compute loss
        
        # Compute argmax of predictions to get accuracy
        _, predicted = torch.max(out.data, 1)
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))
        return loss.cpu().data.numpy(), accuracy
    
    def save_model(self, model_save_name):
        """
        Save the network parameter state, current best accuracy and epoch.
        :param model_save_name: filename to be used for saving this model
        """
        self.state['experiment_state'] = self.state_dict()
        self.state['best_val_score'] = self.best_val_score
        self.state["epoch"] = self.current_epoch
        save_path = self.experiment_checkpoints / "{}.pth".format(model_save_name)
        torch.save(self.state, save_path)
    
    def load_model(self, model_save_name):
        """
        Load the network parameter state, current best validation score and epoch.
        :param model_save_name: Filename (without .pth) of the saved model
        """
        save_path = self.experiment_checkpoints / "{}.pth".format(model_save_name)
        state = torch.load(save_path)
        self.state = state
        
        self.load_state_dict(state_dict=state['experiment_state'])
        self.best_val_score = state["best_val_score"]
        self.start_epoch = self.current_epoch = state["epoch"] + 1
    
    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from start epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": []}  # initialize a dict to keep the per-epoch metrics
        
        if self.start_epoch == 0:
            self.save_model(model_save_name="init")
        
        for i, epoch in enumerate(range(self.start_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            self.current_epoch = epoch
            
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:
                for x, y in self.train_data:
                    loss, accuracy = self.train_batch(x=x, y=y)
                    current_epoch_losses["train_loss"].append(loss)
                    current_epoch_losses["train_acc"].append(accuracy)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                    pbar_train.update()
            
            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
                for x, y in self.val_data:
                    loss, accuracy = self.eval_batch(x=x, y=y)
                    current_epoch_losses["val_loss"].append(loss)
                    current_epoch_losses["val_acc"].append(accuracy)
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                    pbar_val.update()
            
            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_score:
                self.best_val_score = val_mean_accuracy
            
            for key, value in current_epoch_losses.items():
                # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
                total_losses[key].append(np.mean(value))
            
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv', stats_dict=total_losses,
                            current_epoch=i, continue_from_mode=True if (
                        self.start_epoch != 0 or i > 0) else False)  # save statistics to stats file.
            
            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['model_epoch'] = epoch
            self.save_model(model_save_name="latest")
            if self.save_every_epoch:
                self.save_model(model_save_name=str(self.current_epoch))
        
        return total_losses

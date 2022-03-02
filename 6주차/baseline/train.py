import os
import glob
import timm
#import random
import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import wandb
import pandas as pd
import models.model as callModel
import models.loss as module_loss

from functools import partial
from torch.utils.data import DataLoader
from trainer.trainer import MultiHeadTrainer, Trainer
from dataset.dataset import myDataset, train_transform, val_transform
from utils.util import checkDevice, readFile, getOptimizer, setLog, Parser, setSeed, initwandb
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def main(config, logger):
    device = checkDevice()

    # about model
    model_       = config["net"]["model"]
    freeze      = True if config["net"]["freeze"]=="True" else False
    classes     = int(config["net"]["num_classes"])
    pretrained  = True if config["net"]["pretrained"]=="True" else False
    
    # hyper parameters
    epoch       = int(config["net"]["epoch"])
    batch       = int(config["net"]["batch_size"])
    lr          = float(config["net"]["lr"])
    img_shape   = int(config["net"]["img_shape"])
    
    # for training
    criterion   = config["net"]["criterion"]
    optimizer   = config["net"]["optimizer"]
    
    if config['mode']['mode'] == 'Train':
        # for wandb
        initwandb(config)

        # get model, criterion, optimizer, lr_scheduler
        model = callModel(model_, pretrained, classes, freeze).to(device)
        criterion = getattr(module_loss, criterion)()
        optimizer = getOptimizer(optimizer, model, lr)
        
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.2, 2)

        transform = {"train":train_transform(config), "val":val_transform(config)}

        df_train = pd.read_csv("/opt/ml/baseline/train_df_age58.csv", header=0)
        df_val = pd.read_csv("/opt/ml/baseline/val_df_age58.csv", header=0)

        train_dataset = myDataset(df_train, "ans", transform["train"])
        val_dataset = myDataset(df_val, "ans", transform["val"])

        train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True, drop_last=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size = batch, num_workers=2)
    
        dataloaders = {"train":train_loader, "val":val_loader}
    
        # choose between multihead or singlehead
        if 'multihead' in config["net"]["model"].split("_"):
            trainer = MultiHeadTrainer(logger, config, model, criterion, optimizer, dataloaders, device, epoch)
        else:
            trainer = Trainer(logger, config, model, criterion, optimizer, dataloaders, device, epoch)
        
        trainer.train()

    else:
        df_test = pd.read_csv('/opt/ml/baseline/test.csv', header=0)
        transform = {'test':val_transform(config)}

        test_dataset = myDataset(df_test, 'ans', transform['test'])
        test_loaders = DataLoader(test_dataset, batch_size=batch, num_workers=2)

        dataloader = {'test':test_loaders}
    
        if 'multihead' in config["net"]["model"].split("_"):
            trainer = MultiHeadTrainer(logger, config, model, criterion, optimizer, dataloaders, device, epoch)
        else:
            trainer = Trainer(logger, config, model, criterion, optimizer, dataloaders, device, epoch)
    
        loss, answer = trainer.test()
        
        submission_path = os.path.join('/opt/ml/baseline/submission',model_,'submission.csv')
        info_df = pd.read_csv('/opt/ml/baseline/info.csv')
        info_df['ans'] = answer
        info_df.to_csv(submission_path)
        logger(f'submission file saved at .... {submission_path}' )


if __name__=="__main__":
    setSeed(777)
    config = Parser()
    logger = setLog(config)
    
    main(config, logger)







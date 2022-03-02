import argparse
import configparser
import os
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb
import pandas as pd

from trainer.trainer import MultiHeadTrainer, seed_everything
from trainer.trainer import Trainer 
import models.model as models
import models.loss as module_loss
from dataset.dataset import myDataset
from utils.util import checkDevice, readFile
from utils.util import getOptimizer

import timm

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler





config_all_keys = ["net", "path", "project"]


def main(config):
    seed_everything(777)
    device = checkDevice()

    # about model
    model_name = config["net"]["model"]
    num_classes = int(config["net"]["num_classes"])
    pretrained = True if config["net"]["pretrained"]=="True" else False
    freeze = True if config["net"]["freeze"]=="True" else False
    
    # hyper parameters
    epoch = int(config["net"]["epoch"])
    batch_size = int(config["net"]["batch_size"])
    lr = float(config["net"]["lr"])
    img_shape = int(config["net"]["img_shape"])
    
    # for training
    criterion = config["net"]["criterion"]
    optimizer = config["net"]["optimizer"]
    
    
    config_ = {"epochs":epoch, "batch_size": batch_size, "learning_rate": lr}
    
    wandb.init(project=config["project"]["name"], config = config_, entity="haymrpig")
    wandb.run.name = config["project"]["run_name"]

    model = getattr(models, model_name)(pretrained, num_classes, freeze)
    criterion = getattr(module_loss, criterion)()
    optimizer = getOptimizer(optimizer, model, lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.2, 2)

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((img_shape,img_shape)),
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
            transforms.Resize((img_shape,img_shape)),
        ])


    transform = {"train" : train_transform, "val" : val_transform}

    df_train = pd.read_csv("/opt/ml/baseline/train_df_age58.csv", header=0)
    df_val = pd.read_csv("/opt/ml/baseline/val_df_age58.csv", header=0)

    train_dataset = myDataset(df_train, "ans", transform["train"])
    val_dataset = myDataset(df_val, "ans", transform["val"])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=1)
    
    dataloaders = {"train":train_loader, "val":val_loader}
    if config["net"]["model"] == "efficientnet_b3_pruned_multihead":
        trainer = MultiHeadTrainer(model, criterion, optimizer, config, dataloaders, device, epoch, lr_scheduler)
    else:
        trainer = Trainer(model, criterion, optimizer, config, dataloaders, device, epoch, lr_scheduler)
    
    trainer.train()


if __name__=="__main__":
    args = argparse.ArgumentParser(description="baseline code")
    args.add_argument("--config", default=os.environ.get('CONFIG_FILE', '/opt/ml/baseline/config.cfg'), type=str,
                    help="config file path (default : None)")
    
    args = args.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)







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

from trainer.trainer import Trainer 
import models.model as models
import models.loss as module_loss
from dataset.dataset import myDataset
from utils.util import checkDevice, readFile
config_all_keys = ["net", "path", "project"]

def age_group(x):
    if x == ">= 30 and < 60":
        return 1
    elif x=="< 30":
        return 0
    else:
        return 2

def gender(x):
    if x=="female":
        return 1
    else:
        return 0

def mask_type(x):
    if x=="incorrect":
        return 1
    elif x=="wear":
        return 0
    else:
        return 2
def add_path(x):
    return os.path.join("/opt/ml/input/data/train/images", x)

def main(config):
    device = checkDevice()

    model_name = config["net"]["model"]
    num_classes = int(config["net"]["num_classes"])
    pretrained = True if config["net"]["pretrained"]=="True" else False
    freeze = True if config["net"]["freeze"]=="True" else False
    epoch = int(config["net"]["epoch"])
    batch_size = int(config["net"]["batch_size"])
    lr = float(config["net"]["lr"])

    config_ = {"epochs":epoch, "batch_size": batch_size, "learning_rate": lr}
    wandb.init(project=config["project"]["name"], config = config_, entity="haymrpig")
    wandb.run.name = config["net"]["model"]

    model = getattr(models, config["net"]["model"])(pretrained, num_classes, freeze)
    criterion = getattr(module_loss, config["net"]["criterion"])()
    optimizer = getattr(torch.optim, config["net"]["optimizer"])(model.parameters(), lr)


    df = readFile(config["path"]["train_data"], config["path"]["train_data_name"], int(config["path"]["train_header"]))
    df["age_group"] = df["age_group"].apply(age_group)
    df["gender"] = df["gender"].apply(gender)
    df["mask_type"] = df["mask_type"].apply(mask_type)
    df["fname"] = df["fname"].apply(add_path)
    
    df_test = readFile(config["path"]["test_data"], config["path"]["test_data_name"], int(config["path"]["test_header"]))
    df_test["fname"]="/opt/ml/input/data/eval/images/" + df_test["ImageID"]

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((384,512)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384,512)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384,512)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])

    transform = {"train" : train_transform, "val" : val_transform, "test" : test_transform}
    total_images = len(df)
    split_ratio = 0.8
    train_index = int(len(df)*0.8)
    df_train = df.loc[:train_index]
    df_val = df.loc[train_index+1:]


    train_dataset = myDataset(df_train, "ans", transform["train"])
    val_dataset = myDataset(df_val, "ans", transform["val"])
    test_dataset = myDataset(df_test, "ans",transform["test"])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=1) 
    
    dataloaders = {"train":train_loader, "val":val_loader, "test":test_loader}
    trainer = Trainer(model, criterion, optimizer, config, dataloaders, device, epoch)
    
    trainer.train()


if __name__=="__main__":
    args = argparse.ArgumentParser(description="baseline code")
    args.add_argument("--config", default=os.environ.get('CONFIG_FILE', '/opt/ml/baseline/config.cfg'), type=str,
                    help="config file path (default : None)")
    
    args = args.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)







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

from trainer.trainer import seed_everything
from trainer.trainer import MultiHeadTrainer 
import models.model as models
import models.loss as module_loss
from dataset.dataset import myDataset
from utils.util import checkDevice, readFile, saveDataFrame

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

    model = getattr(models, config["net"]["model"])(pretrained, num_classes, freeze)
    model.load_state_dict(torch.load("/opt/ml/baseline/experiments/efficientnet_b3_pruned_multihead13/batch64_focalloss_imgshape300_ReduceLROnPlateau_G/epoch_3_loss_0.126746_acc_77.496038.pt"))
    criterion = getattr(module_loss, config["net"]["criterion"])()
    optimizer = getattr(torch.optim, config["net"]["optimizer"])(model.parameters(), lr)

    df_test = readFile(config["path"]["test_data"], config["path"]["test_data_name"], int(config["path"]["test_header"]))
    df_test["fname"]="/opt/ml/input/data/eval/images/" + df_test["ImageID"]
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((300,300)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])

    transform = {"test" : test_transform}
    test_dataset = myDataset(df_test, "ans",transform["test"])
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=1) 
    dataloaders = {"test":test_loader}

    trainer = MultiHeadTrainer(model, criterion, optimizer, config, dataloaders, device, epoch)

    avg_loss, answer = trainer.test()

    saveDataFrame(answer, config, df_test)



if __name__=="__main__":
    args = argparse.ArgumentParser(description="baseline code")
    args.add_argument("--config", default=os.environ.get('CONFIG_FILE', '/opt/ml/baseline/config.cfg'), type=str,
                    help="config file path (default : None)")
    args = args.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import timm
from timm.optim.optim_factory import create_optimizer
from types import SimpleNamespace

from torch.utils.data import DataLoader

from trainer.trainer import seed_everything
import models.loss as module_loss
from utils.util import checkDevice, readFile
from dataset.dataset import myDataset
import models.model as models

from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.metrics import f1_score

import pandas as pd

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

def train_cifar(config, checkpoint_dir=None, data_dir=None):
    seed_everything(777)

    #net = timm.create_model('efficientnet_b3_pruned', num_classes=18, pretrained=True)
    net = getattr(models, "efficientnet_b3_pruned_multihead")(True)
    device = checkDevice()
    net.to(device)

    criterion = getattr(module_loss, "FocalLoss")()

    args_opt = SimpleNamespace()
    args_opt.weight_decay = 0
    args_opt.lr = config["lr"]
    args_opt.opt = config["optimizer"] #'lookahead_adam' to use `lookahead`
    args_opt.momentum = 0.9
    optimizer = create_optimizer(args_opt, net)

    #config["lr"]
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    
    #img_shape = int(config["image_shape"])
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((300,300)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((300,300)),
            transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
        ])


    transform = {"train" : train_transform, "val" : val_transform}
    df_train = pd.read_csv("/opt/ml/baseline/train_df.csv", header=0)
    df_val = pd.read_csv("/opt/ml/baseline/val_df.csv", header=0)


    train_dataset = myDataset(df_train, "ans", transform["train"])
    val_dataset = myDataset(df_val, "ans", transform["val"])

    trainloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, drop_last=True, num_workers=1)
    valloader = DataLoader(val_dataset, batch_size = config["batch_size"], num_workers=1)
    #int(config["batch_size"])
    

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 200 == 199:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        all_preds, all_labels =[], []

        for i, data in enumerate(valloader, 1):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

                all_preds.append(predicted.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        f1 = f1_score(all_labels, all_preds, average='macro')

        #with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(f1score = f1, loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    
    data_dir = os.path.abspath("./data")
    config = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        #"image_shape" : tune.choice([128, 224, 256, 512]),
        "optimizer" : tune.choice(["Adam", "AdamP"])
    }
    scheduler = ASHAScheduler(
        metric="f1score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        #parameter_columns=["lr", "batch_size", "image_shape"],
        metric_columns=["loss", "f1score", "accuracy", "optimizer", "training_iteration"])
    

    result = tune.run(
        train_cifar,
        resources_per_trial={"cpu": 0, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    #best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #best_trained_model.to(device)

    #best_checkpoint_dir = best_trial.checkpoint.value
    #model_state, optimizer_state = torch.load(os.path.join(
    #    best_checkpoint_dir, "checkpoint"))
    #best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(best_trained_model, device)
    #print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=30, max_num_epochs=10, gpus_per_trial=1)
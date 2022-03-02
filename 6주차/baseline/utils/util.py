import os
import time
import torch
import wandb
import logging
import argparse
import configparser
import numpy as np
import pandas as pd
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer

def checkDevice() -> str :
    """Check if GPU available else Use CPU

    Returns:
        str: "cuda" if GPU is available else "cpu"
    """

    return "cuda" if torch.cuda.is_available() else "cpu"

def readFile(dir_path : str, file_name : str, header : int =None):
    """Read CSV file from directory

    Args:
        dir_path (str): directory path which includes csv file
        file_name (str): target csv file name
        header (int, optional): the index of your csv file column names. Defaults to None.

    Returns:
        DataFrame : DataFrame of the csv file
    """

    full_path = os.path.join(dir_path, file_name)
    return pd.read_csv(full_path, header=header)

def saveDataFrame(answer, config : dict, df_test):
    df_test["ans"] = answer
    df_test["ans"].astype(int)
    fname = config["object"]["type"] + ".csv"
    df_test.to_csv(os.path.join(config['path']["dst_path"],fname), index=False)

def combineAll(config):
    dst_path = config["path"]["dst_path"]
    df_eval_age = pd.read_csv(os.path.join(dst_path,"age_group.csv", header=0))
    df_eval_gender = pd.read_csv(os.path.join(dst_path,"gender.csv", header=0))
    df_eval_mask = pd.read_csv(os.path.join(dst_path,"mask.csv", header=0))

    df_submission = pd.read_csv("/opt/ml/input/data/eval/info.csv", header=0)

    df_submission["ans"] = 6*df_eval_mask + 3*df_eval_gender + df_eval_age
    df_submission.to_csv(os.path.join(dst_path,"submission.csv"), index=False)


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

def makeDatasetDF(config, dir_path, file_name, header=None):
    df = readFile(config["path"]["train_data"], config["path"]["train_data_name"], int(config["path"]["train_header"]))
    df["age_group"] = df["age_group"].apply(age_group)
    df["gender"] = df["gender"].apply(gender)
    df["mask_type"] = df["mask_type"].apply(mask_type)
    df["fname"] = df["fname"].apply(add_path)

    df_sorted = df.sort_values(["gender", "age_group", "path"])
    group = df_sorted.groupby(["gender"])["age_group"].value_counts()

    female_age_0 = df_sorted[(df_sorted["gender"]==1) & (df_sorted["age_group"]==0)]
    female_age_1 = df_sorted[(df_sorted["gender"]==1) & (df_sorted["age_group"]==1)]
    female_age_2 = df_sorted[(df_sorted["gender"]==1) & (df_sorted["age_group"]==2)]

    male_age_0 = df_sorted[(df_sorted["gender"]==0) & (df_sorted["age_group"]==0)]
    male_age_1 = df_sorted[(df_sorted["gender"]==0) & (df_sorted["age_group"]==1)]
    male_age_2 = df_sorted[(df_sorted["gender"]==0) & (df_sorted["age_group"]==2)]

    ratio = 0.8
    train_male_age_0 = male_age_0.iloc[:int(len(male_age_0)*ratio)]
    train_male_age_1 = male_age_1.iloc[:int(len(male_age_1)*ratio)]
    train_male_age_2 = male_age_2.iloc[:int(len(male_age_2)*ratio)]

    train_female_age_0 = female_age_0.iloc[:int(len(female_age_0)*ratio)]
    train_female_age_1 = female_age_1.iloc[:int(len(female_age_1)*ratio)]
    train_female_age_2 = female_age_2.iloc[:int(len(female_age_2)*ratio)]

    val_male_age_0 = male_age_0.iloc[int(len(male_age_0)*ratio):]
    val_male_age_1 = male_age_1.iloc[int(len(male_age_1)*ratio):]
    val_male_age_2 = male_age_2.iloc[int(len(male_age_2)*ratio):]

    val_female_age_0 = female_age_0.iloc[int(len(female_age_0)*ratio):]
    val_female_age_1 = female_age_1.iloc[int(len(female_age_1)*ratio):]
    val_female_age_2 = female_age_2.iloc[int(len(female_age_2)*ratio):]

    train_df = pd.concat([train_male_age_0, train_male_age_1, train_male_age_2, train_female_age_0, train_female_age_1, train_female_age_2], axis=0)
    val_df = pd.concat([val_male_age_0, val_male_age_1, val_male_age_2, val_female_age_0, val_female_age_1, val_female_age_2], axis=0)

    train_df.to_csv("/opt/ml/baseline/train_df.csv", index=False)
    val_df.to_csv("/opt/ml/baseline/val_df.csv", index=False)

    return


def getOptimizer(opt_name, model, lr):
    args = SimpleNamespace()
    args.weight_decay = 0
    args.lr = lr
    args.opt = opt_name
    args.momentum = 0.9

    return create_optimizer(args, model)

def get_log(config):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    model_type  = config['net']['model']
    base_path   = config['path']['base_path']
    full_path   = os.path.join(base_path, 'log')
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    stream_handler = logging.FileHandler(f"full_path/{model_type}_{time.strftime('%m%d-%H-%M-%S')}.txt", 
                                        mode='w', encoding='utf8')
    logger.addHandler(stream_handler)

    return logger

def setLog(config):
    logger = get_log(config)

    mode = config['mode']['mode']
    logger.info(f'\n========={mode} Info=========\n'
                f'Model: {config["net"]["model"]}\n'
                f'Metric: {config["net"]["metric"]}\n'
                f'Criterion: {config["net"]["criterion"]}\n'
                f'Optimizer: {config["net"]["optimizer"]}\n'
                f'Batch size: {config["net"]["batch_size"]}\n'
                f'Epoch: {config["net"]["epoch"]}\n'
                f'Learning rate: {config["net"]["lr"]}\n'
                f'Image shape: {config["net"]["img_shape"]}\n'
                )

    return logger

def Parser():
    args = argparse.ArgumentParser(description="baseline code")
    args.add_argument('-c', '--config', default=os.environ.get('CONFIG_FILE', '/opt/ml/baseline/config.cfg'),
                    help='config file path (default : /opt/ml/baseline/config.cfg)')
    args = args.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)

    return config

def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initwandb(config):
    keys = config['project']['experiment_key'].replace(' ','-').split(',')
    name = ''
    for key in keys:
        name += key
    wandb.init(project=config["project"]["name"], entity='haymrpig', name=name)

    wandb.config.update({'Model': config["net"]["model"],
                'Metric': config["net"]["metric"],
                'Criterion': config["net"]["criterion"],
                'Optimizer': config["net"]["optimizer"],
                'Batch size': config["net"]["batch_size"],
                'Learning rate': config["net"]["lr"],
                'Image shape': config["net"]["img_shape"],
    })



                
    

    

    



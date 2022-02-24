import pandas as pd
import torch
import numpy as np
import os

def checkDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"

def readFile(dir_path, file_name, header=None):
    full_path = os.path.join(dir_path, file_name)
    return pd.read_csv(full_path, header=header)
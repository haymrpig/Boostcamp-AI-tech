import torch
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class myDataset(Dataset):
    def __init__(self, df, class_="ans", transform=None):
        self.df = df
        self.class_ = class_
        self.all_data = df["fname"].values
        self.label = df[class_].values
        self.classes = self.class_name
        self.transform = transform


    def __getitem__(self, idx):
        image = self.all_data[idx]
        label = self.label[idx]

        image = np.array(Image.open(image)) 
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return len(self.label)

    def class_name(self, num):
        if self.class_ == "gender":
            return self.decode_gender(num)

        elif self.class_== "age_group":
            return self.decode_age(num)

        elif self.class_ == "mask":
            return self.decode_mask(num)

        elif self.class_ == "ans":
            return self.decode_all(num)
    
    def decode_gender(self, num):
        if num == 0:
            return "male"
        else:
            return "female"

    def decode_age(self, num):
        if num == 0:
            return "<30"
        elif num==1:
            return "30~60"
        else:
            return "60<"

    def decode_mask(self, num):
        if num==0:
            return "mask"
        elif num==1:
            return "incorrect"
        else:
            return "noMask"

    def decode_all(self, num):
        name = ""
        temp = num//6
        name += self.decode_mask(temp)
        name += "/"
        temp = (num%6)//3
        name += self.decode_gender(temp)
        name += "/"
        temp = (num%6)%3
        name += self.decode_age(temp)
        return name


def train_transform(config):
    
    return A.Compose([
                    A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
                    A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(width=config["img_shape"].split(',')[0], height=config["img_shape"].split(',')[1]),
                    A.Normalize(
                        mean=[0.56, 0.524, 0.501],
                        std=[0.258, 0.265, 0.267],
                        max_pixel_value=255.0),
                    ToTensorV2()
                    ])

def val_transform(config):
    return A.Compose([
                    A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
                    A.Resize(width=config["img_shape"].split(',')[0], height=config["img_shape"].split(',')[1]),
                    A.Normalize(
                        mean=[0.56, 0.524, 0.501],
                        std=[0.258, 0.265, 0.267],
                        max_pixel_value=255.0),
                    ToTensorV2()
                    ])

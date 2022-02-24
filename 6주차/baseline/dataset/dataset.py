from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class myDataset(Dataset):
    def __init__(self, df, class_="all", transform=None):
        self.df = df
        self.class_ = class_
        self.all_data = list(df["fname"])
        self.data_label = list(df[class_])
        self.classes = self.class_name
        self.len = len(self.all_data)
        self.transform = transform


    def __getitem__(self, idx):
        image = self.all_data[idx]
        label = self.data_label[idx]

        image = Image.open(image) 
        
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return self.len

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
        
        


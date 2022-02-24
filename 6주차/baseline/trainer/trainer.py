import torch
from tqdm import tqdm
import glob
import os
import re
from pathlib import Path
import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, model, criterion, optimizer, config, dataloader, device, epoch, lr_scheduler=None):
        self.config = config
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloader
        self.device = device
        self.epoch = int(epoch)
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler

    
    def train(self):
        self.model.train()
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        path = self.increment_path(self.config["path"]["save_dir"])
        os.makedirs(path)
        
        for epoch in range(self.epoch):
            best_loss = 9999999
            for i, (imgs, labels) in enumerate(tqdm(self.dataloaders["train"])):
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                outputs = self.model(imgs)  
                
                loss = self.criterion(outputs, labels)     

                self.optimizer.zero_grad()            
                loss.backward()            
                self.optimizer.step()            

                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax).float().mean()

            avrg_loss, acc = self.validation(epoch + 1)
            if avrg_loss < best_loss:
                print('>>>>>>>>> Best performance at epoch: {}'.format(epoch + 1))
                print('>>>>>>>>> Save model in', path)
                best_loss = avrg_loss
                torch.save(self.model.state_dict(), f"{path}/epoch_{epoch}_loss_{best_loss}_acc_{acc}.pt")
            wandb.log({'accuracy':acc, "loss":avrg_loss})

    def validation(self, epoch):
        print(f'>>>>>>>>> start validation....wait......')
        self.model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0
            cnt = 0
            for i, (imgs, labels) in enumerate(self.dataloaders["val"]):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total += imgs.size(0)
                _, argmax = torch.max(outputs, 1)
                correct += (labels == argmax).sum().item()
                total_loss += loss.item()
                cnt += 1
            avrg_loss = total_loss / cnt
            acc = correct / total * 100
            print('>>>>>>>>> Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, acc, avrg_loss))
        self.model.train()
        return avrg_loss, acc 

    def increment_path(self, path):
        path = Path(path)
        if not path.exists():
            return str(path)
        else:
            dirs = glob.glob(f"{path}*")
            matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{path}{n}"
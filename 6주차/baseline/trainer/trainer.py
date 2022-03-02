import os
import re
import glob
import time
import wandb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, logger, config, model, criterion, optimizer, dataloader, device, epoch, lr_scheduler=None):
        self.logger = logger
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloader
        self.device = device
        self.epoch = epoch
        self.lr_scheduler = lr_scheduler

    
    def train(self):
        
        time_stamp = '_'.join(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()).split(' '))
        path = os.path.join(self.config['path']['base_path'],self.config["path"]["save_dir"], time_stamp)
        os.makedirs(path, exist_ok=True)

        self.model.train()

        best_f1 = 0
        best_acc = 0
        best_loss = 9999999

        early_stop_counter = 0
        early_stop_limit = 15

        for epoch in tqdm(range(self.epoch), total=self.epoch):
            cnt = 0
            correct = 0
            total_loss = 0
            num_samples = 0
            for i, (imgs, labels) in enumerate(tqdm(self.dataloaders['train'], total=len(self.dataloaders['train']))):
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                outputs = self.model(imgs)
                _, argmax = torch.max(outputs, 1)
                correct += (labels == argmax).sum().item()
                
                loss = self.criterion(outputs, labels)     
                self.optimizer.zero_grad()            
                loss.backward()            
                self.optimizer.step()            
                
                total_loss += loss.item()
                num_samples += imgs.size(0)
                cnt += 1

            train_avrg_loss = total_loss / cnt
            train_acc = correct / num_samples * 100

            val_loss, val_acc, f1 = self.validation(epoch + 1)
            torch.save(self.model.state_dict(), f"{path}/epoch_{epoch}_loss_{val_loss:.3f}_acc_{val_acc:.3f}_f1_{f1:.3f}.pt")
            # 모든 모델 저장하기 위함

            lr = get_lr(self.optimizer)
            wandb.log({'val_accuracy':val_acc, "val_loss":val_loss, 'val_fl_score' :f1, 'train_accuracy' : train_acc, "train_loss":train_avrg_loss, "lr":lr})

            if best_f1 < f1:
                trigger_time = 0
                self.logger.info(f'Val F1 improved from {best_f1:.3f} -> {f1:.3f}')
                self.logger.info(f"Save model in {path} as {self.config['net']['model']}best_model.pt....")
                wandb.run.summary["Best F1"] = f1

                best_f1 = f1
                best_loss = val_loss
                best_acc = val_acc

                torch.save(self.model.state_dict(), f"{path}/{self.config['net']['model']}_best_model.pt")
                early_stop_counter = 0
            else:
                early_stop_counter+=1
                self.logger.info(f'Val F1 did not improved from {best_f1:.3f}.. early stopping counter {early_stop_counter} / {early_stop_limit}')

                if early_stop_counter > early_stop_limit:
                    self.logger.info(f'Early stopped!!!!!!!!!!!!!!!')
                    break

            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            

    def validation(self, epoch):
        self.logger.info(f'Start validation....wait.....')

        self.model.eval()
        with torch.no_grad():
            cnt = 0
            correct = 0
            total_loss = 0
            num_samples = 0
            all_preds, all_labels =[], []
            for i, (imgs, labels) in enumerate(self.dataloaders["val"]):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                _, argmax = torch.max(outputs, 1)
                correct += (labels == argmax).sum().item()

                num_samples += imgs.size(0)
                total_loss += loss.item()
                cnt += 1
                all_preds.append(argmax.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            avrg_loss = total_loss / cnt
            acc = correct / num_samples * 100

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            f1 = f1_score(all_labels, all_preds, average='macro')

            self.logger(f'Validation #{epoch}  Accuracy: {acc:.2f}%  Average Loss: {avrg_loss:.4f} f1 score: {f1:.4f}')
        
        self.model.train()
        
        return avrg_loss, acc, f1

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

    def test(self):
        answer=np.array([])

        self.model.eval()
        with torch.no_grad():
            cnt = 0
            total_loss = 0
            self.logger.info(f'Inference start!!!!')
            for i, (imgs, labels) in enumerate(tqdm(self.dataloaders["test"])):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                _, argmax = torch.max(outputs, 1)
                answer = np.append(answer, np.array(argmax.cpu().detach()))
                
                total_loss += loss.item()
                cnt += 1
            avrg_loss = total_loss / cnt
            self.logger(f'Inference End...test Average Loss: {avrg_loss:.4f}')

        return avrg_loss, answer


class MultiHeadTrainer:
    def __init__(self, model, criterion, optimizer, config, dataloader, device, epoch, lr_scheduler=None):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloader
        self.device = device
        self.epoch = int(epoch)
        self.lr_scheduler = lr_scheduler

    
    def train(self):
        self.model.train()
        wandb.watch(self.model, self.criterion, log="all", log_freq=60)
        path = self.increment_path(self.config["path"]["save_dir"])
        path = path + "/" + self.config["project"]["experiment_name"]
        os.makedirs(path)
        
        best_loss = 9999999
        trigger_time = 0
        limit = 30

        for epoch in range(self.epoch):
            total = 0
            correct = 0
            total_loss = 0
            cnt = 0
            for i, (imgs, labels) in enumerate(tqdm(self.dataloaders["train"])):
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                mask_labels = labels//6
                gender_labels = (labels%6)//3
                age_labels = (labels%6)%3

                age_out, mask_out, gender_out = self.model(imgs)
                age_loss = self.criterion(age_out, age_labels)     
                mask_loss = self.criterion(mask_out, mask_labels)
                gender_loss = self.criterion(gender_out, gender_labels)
                loss = age_loss + mask_loss/3 + gender_loss/2

                self.optimizer.zero_grad()            
                loss.backward()            
                self.optimizer.step()            

                _, age_argmax = torch.max(age_out, 1)
                _, mask_argmax = torch.max(mask_out, 1)
                _, gender_argmax = torch.max(gender_out, 1)
                argmax = 6*mask_argmax + 3*gender_argmax + age_argmax
                accuracy = (labels == argmax).float().mean()

                correct += (labels == argmax).sum().item()
                total += imgs.size(0)
                total_loss += loss.item()
                cnt += 1

            
            train_avrg_loss = total_loss / cnt
            train_acc = correct / total * 100
            avrg_loss, acc, f1 = self.validation(epoch + 1)
            lr = get_lr(self.optimizer)

            torch.save(self.model.state_dict(), f"{path}/epoch_{epoch}_loss_{avrg_loss:.6f}_acc_{acc:.6f}.pt")
            # 모든 모델 저장하기 위함

            wandb.log({'val_accuracy':acc, "val_loss":avrg_loss, 'val_fl_score' :f1, 'train_accuracy' : train_acc, "train_loss":train_avrg_loss, "lr":lr})

            if avrg_loss < best_loss:
                trigger_time = 0
                print('>>>>>>>>> Best performance at epoch: {}'.format(epoch + 1))
                print('>>>>>>>>> Save model in', path)
                best_loss = avrg_loss
                torch.save(self.model.state_dict(), f"{path}/best_model.pt")
            else:
                trigger_time+=1
                print(f">>>>>>>>> earlystopping 게이지 {trigger_time} / {limit}")
                if trigger_time > limit:
                    print("early stopped!!!")
                    return
            if self.lr_scheduler:
                self.lr_scheduler.step(avrg_loss)

            

    def validation(self, epoch):
        print(f'>>>>>>>>> start validation....wait......')
        self.model.eval()
        
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0
            cnt = 0
            all_preds, all_labels =[], []
            for i, (imgs, labels) in enumerate(self.dataloaders["val"]):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
    
                mask_labels = labels//6
                gender_labels = (labels%6)//3
                age_labels = (labels%6)%3

                age_out, mask_out, gender_out = self.model(imgs)
                age_loss = self.criterion(age_out, age_labels)     
                mask_loss = self.criterion(mask_out, mask_labels)
                gender_loss = self.criterion(gender_out, gender_labels)
                loss = age_loss + mask_loss/3 + gender_loss/2

                _, age_argmax = torch.max(age_out, 1)
                _, mask_argmax = torch.max(mask_out, 1)
                _, gender_argmax = torch.max(gender_out, 1)
                argmax = 6*mask_argmax + 3*gender_argmax + age_argmax
                correct += (labels == argmax).sum().item()

                total += imgs.size(0)
                total_loss += loss.item()
                cnt += 1
                all_preds.append(argmax.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            avrg_loss = total_loss / cnt
            acc = correct / total * 100
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # save plt
            df_preds = pd.DataFrame(data=all_preds, columns=["preds"])
            df_labels = pd.DataFrame(data=all_labels, columns=["label_answer"])
            new_df = pd.concat([df_preds, df_labels], axis=1)
            data = new_df.groupby(["label_answer"])["preds"].value_counts().unstack()
            data = data.fillna(0)
            data= data.astype(int)
            fig, ax = plt.subplots(1,1,dpi=120)
            ax = sns.heatmap(data,annot=True, fmt="d")
            fig.savefig(f"/opt/ml/baseline/multihead_58_28_aug-epoch{epoch}.png")

            df_preds.to_csv("/opt/ml/baseline/preds.csv", index=False)
            df_labels.to_csv("/opt/ml/baseline/labels.csv", index=False)
            f1 = f1_score(all_labels, all_preds, average='macro')

            print('>>>>>>>>> Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f} f1 score: {:.4f}'.format(epoch, acc, avrg_loss, f1))
        self.model.train()
        return avrg_loss, acc, f1

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

    def test(self):
        answer=np.array([])
        self.model.eval()

        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0
            cnt = 0
            for i, (imgs, labels) in enumerate(tqdm(self.dataloaders["test"])):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                age_out, mask_out, gender_out = self.model(imgs)
                
                age_loss = self.criterion(age_out, labels)     
                mask_loss = self.criterion(mask_out, labels)
                gender_loss = self.criterion(gender_out, labels)
                loss = 1/2*age_loss + 1/4*mask_loss + 1/4*gender_loss

                _, age_argmax = torch.max(age_out, 1)
                _, mask_argmax = torch.max(mask_out, 1)
                _, gender_argmax = torch.max(gender_out, 1)
                argmax = 6*mask_argmax + 3*gender_argmax + age_argmax
                answer = np.append(answer, np.array(argmax.cpu().detach()))
                total_loss += loss.item()
                cnt += 1
            avrg_loss = total_loss / cnt
            print('test #{} Average Loss: {:.4f}'.format(self.epoch, avrg_loss))

        return avrg_loss, answer

    
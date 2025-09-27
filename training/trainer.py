import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from .metrics import calculate_mcc

class GlacierTrainer:
    def __init__(self, model, optimizer, loss_fn, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.val_mccs = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        self.train_losses.append(epoch_loss)
        return epoch_loss
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                running_loss += loss.item()
                
                preds = torch.sigmoid(output) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        self.val_losses.append(epoch_loss)
        
        # Calculate MCC
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        mcc = calculate_mcc(all_preds, all_targets)
        self.val_mccs.append(mcc)
        
        return epoch_loss, mcc
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mccs': self.val_mccs
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_mccs = checkpoint['val_mccs']
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .metrics import calculate_mcc
import os
from datetime import datetime

class GlacierTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Create logging directory
        log_dir = os.path.join(config['paths']['logs_dir'], 
                              f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter(log_dir)
        
        self.best_mcc = -1.0
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Convert to binary predictions
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(masks.cpu().numpy())
        
        # Calculate MCC
        mcc = calculate_mcc(all_preds, all_targets)
        return val_loss / len(self.val_loader), mcc
    
    def train(self):
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch()
            val_loss, mcc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/MCC', mcc, epoch)
            
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}: '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'MCC: {mcc:.4f}')
            
            # Save best model
            if mcc > self.best_mcc:
                self.best_mcc = mcc
                self.patience_counter = 0
                self.save_model('best_model.pth')
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                print("Early stopping triggered!")
                break
                
        self.writer.close()
        
    def save_model(self, filename):
        model_path = os.path.join(self.config['paths']['model_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mcc': self.best_mcc
        }, model_path)
        
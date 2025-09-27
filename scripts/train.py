import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import optuna
from data.dataset import GlacierDataset
from data.transforms import get_train_transforms, get_val_transforms
from data.utils import load_config, split_dataset
from models.model_factory import create_model
from models.losses import get_loss
from training.trainer import GlacierTrainer
from training.optimizer import get_optimizer
from utils.visualization import plot_training_history

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_transforms = get_train_transforms(config['data']['image_size'])
    val_transforms = get_val_transforms(config['data']['image_size'])
    
    dataset = GlacierDataset(
        root_dir='/kaggle/input/glacer/Train',
        bands=['Band1', 'Band2', 'Band3', 'Band4', 'Band5'],
        label_dir='label',
        transform=train_transforms,
        image_size=config['data']['image_size']
    )
    
    # Split dataset
    train_idx, val_idx = split_dataset(dataset, train_size=config['data']['train_split'])
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create model
    model_config = config['model_configs'][config['model']['name']]
    model = create_model(config['model']['name'], model_config)
    model = model.to(device)
    
    # Create loss function and optimizer
    loss_fn = get_loss(config['training']['loss'])
    optimizer = get_optimizer(config['training']['optimizer'], model.parameters(), config['training']['learning_rate'])
    
    # Create trainer
    trainer = GlacierTrainer(model, optimizer, loss_fn, device)
    
    # Training loop
    best_mcc = 0.0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_mcc = trainer.validate_epoch(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}")
        
        # Save best model
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            trainer.save_checkpoint('best_model.pth')
            print(f"New best model saved with MCC: {best_mcc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print("Early stopping triggered")
            break
    
    # Plot training history
    plot_training_history(trainer, 'training_history.png')
    
    print(f"Training completed. Best MCC: {best_mcc:.4f}")

if __name__ == "__main__":
    main()
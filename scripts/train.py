import torch
import yaml
from data.dataset import DataLoaderFactory
from models.model_factory import ModelFactory
from models.losses import CombinedLoss
from training.trainer import GlacierTrainer
from training.optimizer import get_optimizer, get_scheduler

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    loader_factory = DataLoaderFactory(config)
    train_loader, val_loader = loader_factory.create_loaders(config['paths']['data_dir'])
    
    # Create model
    model_config = MODEL_CONFIGS[config['model']['name']]
    model_config.input_channels = config['data']['channels']
    model = ModelFactory.create_model(model_config)
    model = model.to(device)
    
    # Create loss function, optimizer, and scheduler
    criterion = CombinedLoss()
    optimizer = get_optimizer(model, 'adam', config['training']['learning_rate'])
    scheduler = get_scheduler(optimizer, 'reduce_lr')
    
    # Create trainer and start training
    trainer = GlacierTrainer(model, train_loader, val_loader, criterion,
                           optimizer, scheduler, device, config)
    trainer.train()
    
    print(f"Training completed. Best MCC: {trainer.best_mcc:.4f}")

if __name__ == "__main__":
    main()
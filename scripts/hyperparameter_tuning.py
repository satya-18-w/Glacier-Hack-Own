import optuna
import torch
import yaml
from data.dataset import DataLoaderFactory
from models.model_factory import ModelFactory
from models.losses import CombinedLoss, FocalLoss, DiceLoss
from training.trainer import GlacierTrainer
from training.optimizer import get_optimizer, get_scheduler

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    model_name = trial.suggest_categorical('model_name', 
                                          ['unet', 'attention_unet', 'deeplabv3', 'fpn'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    loss_fn = trial.suggest_categorical('loss_fn', ['bce', 'dice', 'combined', 'focal'])
    
    # Update config
    config['data']['batch_size'] = batch_size
    config['model']['name'] = model_name
    config['training']['learning_rate'] = lr
    
    # Create data loaders
    loader_factory = DataLoaderFactory(config)
    train_loader, val_loader = loader_factory.create_loaders(config['paths']['data_dir'])
    
    # Create model
    model_config = MODEL_CONFIGS[model_name]
    model_config.input_channels = config['data']['channels']
    model = ModelFactory.create_model(model_config)
    model = model.to(device)
    
    # Create loss function
    if loss_fn == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_fn == 'dice':
        criterion = DiceLoss()
    elif loss_fn == 'combined':
        criterion = CombinedLoss()
    elif loss_fn == 'focal':
        criterion = FocalLoss()
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = get_scheduler(optimizer, 'reduce_lr')
    
    # Train model
    trainer = GlacierTrainer(model, train_loader, val_loader, criterion,
                           optimizer, scheduler, device, config)
    trainer.train()
    
    return trainer.best_mcc

def run_hyperparameter_tuning(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config['hyperparameter_tuning']['n_trials'],
                  timeout=config['hyperparameter_tuning']['timeout'])
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_params = run_hyperparameter_tuning('config/config.yaml')
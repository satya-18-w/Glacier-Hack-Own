import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from data.fixed_dataset import FixedDataLoaderFactory
from models.model_factory import create_model
from training.trainer import GlacierTrainer

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Suggest hyperparameters
    config['training']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    config['model']['encoder'] = trial.suggest_categorical('encoder', ['resnet18', 'resnet34', 'resnet50'])
    config['data']['batch_size'] = trial.suggest_categorical('batch_size', [4, 8, 16])
    config['training']['epochs'] = 10 # Reduced for faster tuning

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    data_loader_factory = FixedDataLoaderFactory(config)
    train_loader, val_loader = data_loader_factory.create_loaders()

    # Create model
    model_config = config['model']
    model = create_model(model_config['name'], model_config)
    model.to(device)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()

    # Create trainer
    trainer = GlacierTrainer(model, optimizer, loss_fn, device)

    # Train and evaluate
    best_mcc = 0
    for epoch in range(config['training']['epochs']):
        trainer.train_epoch(train_loader)
        val_loss, val_mcc = trainer.validate_epoch(val_loader)
        if val_mcc > best_mcc:
            best_mcc = val_mcc
        
        trial.report(best_mcc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_mcc

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

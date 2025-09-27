import optuna
import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import GlacierDataset
from data.transforms import get_train_transforms, get_val_transforms
from data.utils import split_dataset
from models.model_factory import create_model
from models.losses import get_loss
from training.trainer import GlacierTrainer
from training.optimizer import get_optimizer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    model_name = trial.suggest_categorical('model', ['unet', 'attention_unet', 'deeplabv3plus'])
    loss_name = trial.suggest_categorical('loss', ['bce', 'dice', 'bce_dice', 'focal'])
    
    # Create dataset
    dataset = GlacierDataset(
        root_dir='/kaggle/input/glacer/Train',
        bands=['Band1', 'Band2', 'Band3', 'Band4', 'Band5'],
        label_dir='label',
        transform=get_train_transforms(),
        image_size=(256, 256)
    )
    
    train_idx, val_idx = split_dataset(dataset)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, {'in_channels': 5, 'classes': 1})
    model = model.to(device)
    
    # Create loss and optimizer
    loss_fn = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    
    # Train for a few epochs
    trainer = GlacierTrainer(model, optimizer, loss_fn, device)
    
    best_mcc = 0
    for epoch in range(10):  # Short training for tuning
        trainer.train_epoch(train_loader)
        _, val_mcc = trainer.validate_epoch(val_loader)
        
        if val_mcc > best_mcc:
            best_mcc = val_mcc
        
        trial.report(val_mcc, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_mcc

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=3600)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
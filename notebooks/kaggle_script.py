#%% [markdown]
# Glacier Segmentation - Kaggle Training Notebook

# %% [code]
# All necessary imports
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import optuna
import segmentation_models_pytorch as smp

# %% [markdown]
# ## 1. Configuration

# %% [code]
# Configuration for the training
config_str = """
data:
  image_size: [256, 256]
  channels: 5
  batch_size: 8
  train_split: 0.8
  val_split: 0.2
  spectral_bands: ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']

model:
  name: 'unet'
  encoder: 'resnet34'
  encoder_weights: 'imagenet' # Using pretrained weights
  activation: 'sigmoid'

training:
  epochs: 50
  learning_rate: 0.001
  patience: 10
  min_delta: 0.001
  use_amp: true

paths:
  data_dir: '/kaggle/input/glacier-hack-data' # Kaggle data path
  model_dir: '/kaggle/working/models'
  logs_dir: '/kaggle/working/logs'
"""

config = yaml.safe_load(config_str)

# Create directories if they don't exist
os.makedirs(config['paths']['model_dir'], exist_ok=True)
os.makedirs(config['paths']['logs_dir'], exist_ok=True)

# %% [markdown]
# ## 2. Data Loading

# %% [code]
class FixedGlacierDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def load_single_band(self, path):
        """Load single band image"""
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                return data.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def __getitem__(self, idx):
        band_paths = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        bands = []
        for band_path in band_paths:
            band_data = self.load_single_band(band_path)
            bands.append(band_data)
        
        image = np.stack(bands, axis=0)
        
        mask = self.load_single_band(mask_path)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_transposed = image.transpose(1, 2, 0)
            augmented = self.transform(image=image_transposed, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            
        return image, mask.unsqueeze(0)

class FixedDataLoaderFactory:
    def __init__(self, config):
        self.config = config
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0], self.config['data']['image_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0], self.config['data']['image_size'][1]),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),
            ToTensorV2(),
        ])
        
        return {'train': train_transform, 'val': val_transform}
    
    def find_band_files(self, base_path):
        print("=== LOADING DATASET ===")
        band_dirs = {band: os.path.join(base_path, 'Train', band) for band in self.config['data']['spectral_bands']}
        mask_dir = os.path.join(base_path, 'Train', 'label')
        
        existing_bands = {name: path for name, path in band_dirs.items() if os.path.exists(path)}
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory not found: {mask_dir}")
            return [], []
        
        first_band_dir = list(existing_bands.values())[0]
        reference_files = [f for f in os.listdir(first_band_dir) if f.endswith('.tif')]
        
        image_groups = []
        mask_paths = []
        
        for ref_file in reference_files:
            base_name = ref_file.split('.')[0]
            parts = base_name.split('_')
            base_id = '_'.join(parts[-2:])

            band_paths = []
            valid_group = True
            
            for band_name in existing_bands.keys():
                band_dir = existing_bands[band_name]
                pattern_files = [f for f in os.listdir(band_dir) if base_id in f and f.endswith('.tif')]
                
                if pattern_files:
                    band_paths.append(os.path.join(band_dir, pattern_files[0]))
                else:
                    valid_group = False
                    break
            
            mask_pattern_files = [f for f in os.listdir(mask_dir) if base_id in f and f.endswith('.tif')]
            
            if mask_pattern_files and valid_group:
                mask_path = os.path.join(mask_dir, mask_pattern_files[0])
                image_groups.append(band_paths)
                mask_paths.append(mask_path)
        
        print(f"Created dataset with {len(image_groups)} samples")
        return image_groups, mask_paths
    
    def create_loaders(self):
        image_groups, mask_paths = self.find_band_files(self.config['paths']['data_dir'])
        
        if len(image_groups) == 0:
            raise ValueError("No valid image-mask pairs found!")
        
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_groups, mask_paths, 
            test_size=self.config['data']['val_split'],
            random_state=42,
            shuffle=True
        )
        
        train_dataset = FixedGlacierDataset(train_img, train_mask, self.transform['train'], self.config['data']['image_size'])
        val_dataset = FixedGlacierDataset(val_img, val_mask, self.transform['val'], self.config['data']['image_size'])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader

# %% [markdown]
# ## 3. Model Definition

# %% [code]
class GlacierUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=5, classes=1):
        super(GlacierUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    def forward(self, x):
        return self.model(x)

def create_model(model_name, model_config):
    if model_name == "unet":
        return GlacierUNet(
            encoder_name=model_config.get('encoder', 'resnet34'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet'),
            in_channels=model_config.get('channels', 5),
            classes=1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

# %% [markdown]
# ## 4. Metrics

# %% [code]
def calculate_mcc(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).float()

    y_true = y_true.cpu().numpy().flatten()
    y_pred_bin = y_pred_bin.cpu().numpy().flatten()

    tp = ((y_pred_bin == 1) & (y_true == 1)).sum()
    tn = ((y_pred_bin == 0) & (y_true == 0)).sum()
    fp = ((y_pred_bin == 1) & (y_true == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_true == 1)).sum()

    numerator = (tp * tn) - (fp * fn)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if denominator == 0:
        return 0.0

    mcc = numerator / (denominator**0.5)
    return mcc

# %% [markdown]
# ## 5. Trainer

# %% [code]
class GlacierTrainer:
    def __init__(self, model, optimizer, loss_fn, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        for data, target in tqdm(dataloader, desc="Training"):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
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
                
                all_preds.append(output)
                all_targets.append(target)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mcc = calculate_mcc(all_targets, all_preds)
        
        return running_loss / len(dataloader), mcc

# %% [markdown]
# ## 6. Hyperparameter Tuning with Optuna

# %% [code]
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    encoder = trial.suggest_categorical('encoder', ['resnet18', 'resnet34', 'resnet50'])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    # Update config
    temp_config = config.copy()
    temp_config['training']['learning_rate'] = learning_rate
    temp_config['model']['encoder'] = encoder
    temp_config['data']['batch_size'] = batch_size
    temp_config['training']['epochs'] = 10 # Reduced for faster tuning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_factory = FixedDataLoaderFactory(temp_config)
    train_loader, val_loader = data_loader_factory.create_loaders()

    model = create_model(temp_config['model']['name'], temp_config['model'])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=temp_config['training']['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = GlacierTrainer(model, optimizer, loss_fn, device)

    best_mcc = 0
    for epoch in range(temp_config['training']['epochs']):
        trainer.train_epoch(train_loader)
        _, val_mcc = trainer.validate_epoch(val_loader)
        if val_mcc > best_mcc:
            best_mcc = val_mcc
        
        trial.report(best_mcc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_mcc

# %% [code]
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %% [markdown]
# ## 7. Final Training

# %% [code]
# Update config with best params
config['training']['learning_rate'] = trial.params['learning_rate']
config['model']['encoder'] = trial.params['encoder']
config['data']['batch_size'] = trial.params['batch_size']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader_factory = FixedDataLoaderFactory(config)
train_loader, val_loader = data_loader_factory.create_loaders()

model = create_model(config['model']['name'], config['model'])
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
loss_fn = nn.BCEWithLogitsLoss()

trainer = GlacierTrainer(model, optimizer, loss_fn, device)

for epoch in range(config['training']['epochs']):
    print(f"Epoch {epoch+1}/{config['training']['epochs']}")
    train_loss = trainer.train_epoch(train_loader)
    val_loss, val_mcc = trainer.validate_epoch(val_loader)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}")

# Save the final model
final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

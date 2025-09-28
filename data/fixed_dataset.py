import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path

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
                data = src.read(1)  # Read first band
                return data.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def __getitem__(self, idx):
        # Load all 5 bands for this sample
        band_paths = self.image_paths[idx]  # List of 5 band paths
        mask_path = self.mask_paths[idx]    # Single mask path
        
        # Load and stack bands
        bands = []
        for band_path in band_paths:
            band_data = self.load_single_band(band_path)
            bands.append(band_data)
        
        # Stack bands to create 5-channel image
        image = np.stack(bands, axis=0)  # Shape: (5, H, W)
        
        # Load mask
        mask = self.load_single_band(mask_path)
        mask = (mask > 0).astype(np.float32)  # Convert to binary
        
        # Apply transformations
        if self.transform:
            # Transpose to (H, W, C) for albumentations
            image_transposed = image.transpose(1, 2, 0)
            augmented = self.transform(image=image_transposed, mask=mask)
            image = augmented['image']  # Already (C, H, W) from ToTensorV2
            mask = augmented['mask']
        else:
            # Convert to tensor without transform
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            
        return image, mask.unsqueeze(0)  # Add channel dimension to mask

class FixedDataLoaderFactory:
    def __init__(self, config):
        self.config = config
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Get data augmentation transforms"""
        train_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0], 
                    self.config['data']['image_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0],
                    self.config['data']['image_size'][1]),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),
            ToTensorV2(),
        ])
        
        return {'train': train_transform, 'val': val_transform}
    
    def find_band_files(self, base_path):
        """Find and group band files correctly"""
        print("=== LOADING DATASET ===")
        
        # Define band directories based on the actual structure
        band_dirs = {
            'Band1': os.path.join(base_path, 'Train', 'Band1'),
            'Band2': os.path.join(base_path, 'Train', 'Band2'),
            'Band3': os.path.join(base_path, 'Train', 'Band3'),
            'Band4': os.path.join(base_path, 'Train', 'Band4'),
            'Band5': os.path.join(base_path, 'Train', 'Band5')
        }
        
        mask_dir = os.path.join(base_path, 'Train', 'label')
        
        # Check which directories exist
        existing_bands = {}
        for band_name, band_path in band_dirs.items():
            if os.path.exists(band_path):
                existing_bands[band_name] = band_path
                print(f"Found {band_name} directory: {band_path}")
            else:
                print(f"Warning: {band_name} directory not found: {band_path}")
        
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory not found: {mask_dir}")
            return [], []
        
        print(f"Using bands: {list(existing_bands.keys())}")
        
        # Get all files from first band directory to use as reference
        first_band_dir = list(existing_bands.values())[0]
        reference_files = [f for f in os.listdir(first_band_dir) if f.endswith('.tif')]
        
        image_groups = []
        mask_paths = []
        
        for ref_file in reference_files:
            # Extract base name (e.g., "02_07" from "B2_B2_masked_02_07.tif")
            base_name = ref_file.split('.')[0]
            parts = base_name.split('_')
            base_id = '_'.join(parts[-2:])

            # Find corresponding files in all bands
            band_paths = []
            valid_group = True
            
            for band_name in existing_bands.keys():
                band_dir = existing_bands[band_name]
                # Look for file with the same base_id
                pattern_files = [f for f in os.listdir(band_dir) 
                               if base_id in f and f.endswith('.tif')]
                
                if pattern_files:
                    band_paths.append(os.path.join(band_dir, pattern_files[0]))
                else:
                    print(f"Warning: No matching file for {base_id} in {band_name}")
                    valid_group = False
                    break
            
            # Find corresponding mask
            mask_pattern_files = [f for f in os.listdir(mask_dir) 
                                if base_id in f and f.endswith('.tif')]
            
            if mask_pattern_files and valid_group:
                mask_path = os.path.join(mask_dir, mask_pattern_files[0])
                image_groups.append(band_paths)
                mask_paths.append(mask_path)
        
        print(f"Created dataset with {len(image_groups)} samples")
        return image_groups, mask_paths
    
    def create_loaders(self):
        """Create data loaders for training and validation"""
        image_groups, mask_paths = self.find_band_files(self.config['paths']['data_dir'])
        
        if len(image_groups) == 0:
            raise ValueError("No valid image-mask pairs found!")
        
        # Split data
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_groups, mask_paths, 
            test_size=self.config['data']['val_split'],
            random_state=42,
            shuffle=True
        )
        
        print(f"Training samples: {len(train_img)}, Validation samples: {len(val_img)}")
        
        # Create datasets
        train_dataset = FixedGlacierDataset(train_img, train_mask, 
                                          self.transform['train'],
                                          self.config['data']['image_size'])
        val_dataset = FixedGlacierDataset(val_img, val_mask, 
                                        self.transform['val'],
                                        self.config['data']['image_size'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['data']['batch_size'],
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.config['data']['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader
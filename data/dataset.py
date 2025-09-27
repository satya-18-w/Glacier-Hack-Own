import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import cv2

class GlacierDataset(Dataset):
    def __init__(self, root_dir, bands, label_dir, transform=None, image_size=(256, 256)):
        self.root_dir = root_dir
        self.bands = bands
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load all sample names from the first band directory"""
        band1_dir = os.path.join(self.root_dir, self.bands[0])
        if not os.path.exists(band1_dir):
            # Try to find available image files in root directory
            samples = [f for f in os.listdir(self.root_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
            print(f"Found {len(samples)} samples in root directory")
            return samples
        
        samples = [f for f in os.listdir(band1_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
        print(f"Found {len(samples)} samples in {band1_dir}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_band(self, band_path):
        """Load individual band with proper error handling"""
        try:
            if band_path.endswith(('.tif', '.tiff')):
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
            else:
                band_data = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
            
            if band_data is None:
                print(f"Warning: Could not load {band_path}")
                return np.zeros(self.image_size, dtype=np.float32)
            
            # Resize if necessary
            if band_data.shape != self.image_size:
                band_data = cv2.resize(band_data, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            return band_data
        except Exception as e:
            print(f"Error loading {band_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def _get_band_path(self, sample_name, band):
        """Get the correct path for a band, handling different directory structures"""
        band_dir = os.path.join(self.root_dir, band)
        if os.path.exists(band_dir):
            return os.path.join(band_dir, sample_name)
        else:
            # If band directories don't exist, try different naming patterns
            possible_paths = [
                os.path.join(self.root_dir, f"{band}_{sample_name}"),
                os.path.join(self.root_dir, sample_name.replace('.', f'_{band}.')),
                os.path.join(self.root_dir, sample_name)
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return os.path.join(self.root_dir, sample_name)  # Fallback
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # Load all bands
        band_images = []
        for band in self.bands:
            band_path = self._get_band_path(sample_name, band)
            band_data = self._load_band(band_path)
            band_images.append(band_data)
        
        # Stack bands to create 5-channel image
        image = np.stack(band_images, axis=0)  # Shape: (5, H, W)
        image = image.astype(np.float32)
        
        # Normalize each band
        for i in range(image.shape[0]):
            if np.max(image[i]) > 0:
                image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        
        # Load label
        label_path = self._get_band_path(sample_name, self.label_dir)
        try:
            if label_path.endswith(('.tif', '.tiff')):
                with rasterio.open(label_path) as src:
                    label = src.read(1)
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            if label is None:
                label = np.zeros(self.image_size, dtype=np.float32)
            
            if label.shape != self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            label = (label > 0).astype(np.float32)  # Binarize
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            label = np.zeros(self.image_size, dtype=np.float32)
        
        # Apply transformations - FIXED: Handle tensor conversion properly
        if self.transform:
            # Convert to HWC format for albumentations
            image_hwc = image.transpose(1, 2, 0)
            augmented = self.transform(image=image_hwc, mask=label)
            
            # albumentations with ToTensorV2 already converts to tensor
            image = augmented['image']  # Already tensor of shape (5, H, W)
            label = augmented['mask']   # Already tensor of shape (H, W)
            
            # Add channel dimension to label if needed
            if label.dim() == 2:
                label = label.unsqueeze(0)  # Shape: (1, H, W)
        else:
            # Convert to tensors if no transform
            image = torch.from_numpy(image)
            label = torch.from_numpy(label).unsqueeze(0)
        
        return image, label
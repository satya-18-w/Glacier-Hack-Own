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
        samples = [f for f in os.listdir(band1_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
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
            
            # Resize if necessary
            if band_data.shape != self.image_size:
                band_data = cv2.resize(band_data, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            return band_data
        except Exception as e:
            print(f"Error loading {band_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # Load all bands
        band_images = []
        for band in self.bands:
            band_path = os.path.join(self.root_dir, band, sample_name)
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
        label_path = os.path.join(self.root_dir, self.label_dir, sample_name)
        try:
            if label_path.endswith(('.tif', '.tiff')):
                with rasterio.open(label_path) as src:
                    label = src.read(1)
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            if label.shape != self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            label = (label > 0).astype(np.float32)  # Binarize
        except:
            label = np.zeros(self.image_size, dtype=np.float32)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=label)
            image = augmented['image'].transpose(2, 0, 1)
            label = augmented['mask']
        
        return torch.from_numpy(image), torch.from_numpy(label).unsqueeze(0)
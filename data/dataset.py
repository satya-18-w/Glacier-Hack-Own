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
        """Load all sample names by finding common files across bands"""
        # Get files from the first band directory
        first_band_dir = os.path.join(self.root_dir, self.bands[0])
        if not os.path.exists(first_band_dir):
            # If band directories don't exist, look for files in root
            all_files = [f for f in os.listdir(self.root_dir) 
                        if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
            print(f"Found {len(all_files)} files in root directory")
            return all_files
        
        first_band_files = set([f for f in os.listdir(first_band_dir) 
                               if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
        
        # Find common files across all bands
        common_files = first_band_files.copy()
        
        for band in self.bands[1:]:
            band_dir = os.path.join(self.root_dir, band)
            if os.path.exists(band_dir):
                band_files = set([f for f in os.listdir(band_dir) 
                                 if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
                common_files = common_files.intersection(band_files)
        
        # Also check label directory
        label_dir_path = os.path.join(self.root_dir, self.label_dir)
        if os.path.exists(label_dir_path):
            label_files = set([f for f in os.listdir(label_dir_path) 
                             if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
            common_files = common_files.intersection(label_files)
        
        samples = list(common_files)
        samples.sort()
        print(f"Found {len(samples)} common samples across all bands and labels")
        
        if len(samples) == 0:
            print("Warning: No common files found. Using files from first band only.")
            samples = list(first_band_files)
            samples.sort()
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_band(self, band_path):
        """Load individual band with proper error handling"""
        try:
            if not os.path.exists(band_path):
                print(f"File not found: {band_path}")
                return np.zeros(self.image_size, dtype=np.float32)
                
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
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        # print(f"Loading sample: {sample_name}")  # Debug
        
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
            if not os.path.exists(label_path):
                print(f"Label not found: {label_path}")
                label = np.zeros(self.image_size, dtype=np.float32)
            else:
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
        
        # Apply transformations - FIXED VERSION
        if self.transform:
            # Convert to HWC format for albumentations (numpy array)
            image_hwc = image.transpose(1, 2, 0)  # This is numpy operation, works on arrays
            
            # Apply albumentations
            augmented = self.transform(image=image_hwc, mask=label)
            
            # albumentations with ToTensorV2 converts to tensors automatically
            image_tensor = augmented['image']  # Shape: (5, H, W) - already tensor
            label_tensor = augmented['mask']   # Shape: (H, W) - already tensor
            
            # Ensure label has channel dimension
            if label_tensor.dim() == 2:
                label_tensor = label_tensor.unsqueeze(0)  # Shape: (1, H, W)
                
            return image_tensor, label_tensor
        else:
            # Convert numpy arrays to tensors
            image_tensor = torch.from_numpy(image)
            label_tensor = torch.from_numpy(label).unsqueeze(0)
            return image_tensor, label_tensor
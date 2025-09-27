import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import cv2
import re

class GlacierDataset(Dataset):
    def __init__(self, root_dir, bands, label_dir, transform=None, image_size=(256, 256)):
        self.root_dir = root_dir
        self.bands = bands
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.samples = self._load_samples()
    
    def _find_matching_files(self, base_file, target_dir):
        """Find files in target directory that match the pattern of base_file"""
        if not os.path.exists(target_dir):
            return []
            
        # Extract the numeric pattern (e.g., "08_12" from "B2_B2_masked_08_12.tif")
        pattern = re.search(r'(\d+_\d+)\.', base_file)
        if pattern:
            pattern_str = pattern.group(1)
            # Look for files with the same numeric pattern
            matching_files = [f for f in os.listdir(target_dir) if pattern_str in f]
            return matching_files
        return []
    
    def _load_samples(self):
        """Load samples by finding matching files across all bands"""
        print("Scanning for matching files across bands...")
        
        # Start with Band2 as reference (it seems to have the most files)
        ref_band = 'Band2'
        ref_dir = os.path.join(self.root_dir, ref_band)
        
        if not os.path.exists(ref_dir):
            # Try any available band
            for band in self.bands:
                ref_dir = os.path.join(self.root_dir, band)
                if os.path.exists(ref_dir):
                    ref_band = band
                    break
        
        if not os.path.exists(ref_dir):
            print("No band directories found!")
            return []
        
        ref_files = [f for f in os.listdir(ref_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
        ref_files.sort()
        print(f"Found {len(ref_files)} files in {ref_band}")
        
        # For each file in reference band, find matching files in other bands
        valid_samples = []
        
        for ref_file in ref_files:
            # Extract the unique identifier (last part before extension)
            file_parts = ref_file.split('_')
            if len(file_parts) >= 2:
                # Try different patterns to match
                possible_patterns = [
                    '_'.join(file_parts[-3:]),  # B2_masked_08_12
                    '_'.join(file_parts[-2:]),  # masked_08_12
                    file_parts[-1].split('.')[0]  # 12 (last number)
                ]
            else:
                possible_patterns = [ref_file]
            
            # Check if this file has matches in all required bands
            has_all_bands = True
            band_files = {ref_band: ref_file}
            
            for band in self.bands:
                if band == ref_band:
                    continue
                    
                band_dir = os.path.join(self.root_dir, band)
                if not os.path.exists(band_dir):
                    print(f"Band directory {band} not found")
                    has_all_bands = False
                    break
                
                # Look for matching file
                matching_files = []
                for pattern in possible_patterns:
                    matching_files = [f for f in os.listdir(band_dir) if pattern in f]
                    if matching_files:
                        break
                
                if matching_files:
                    band_files[band] = matching_files[0]  # Take first match
                else:
                    print(f"No match found for {ref_file} in {band}")
                    has_all_bands = False
                    break
            
            # Check label directory
            label_dir_path = os.path.join(self.root_dir, self.label_dir)
            if os.path.exists(label_dir_path):
                matching_labels = []
                for pattern in possible_patterns:
                    matching_labels = [f for f in os.listdir(label_dir_path) if pattern in f]
                    if matching_labels:
                        break
                
                if matching_labels:
                    band_files['label'] = matching_labels[0]
                else:
                    print(f"No label match found for {ref_file}")
                    has_all_bands = False
            else:
                print(f"Label directory {label_dir_path} not found")
                has_all_bands = False
            
            if has_all_bands:
                valid_samples.append(band_files)
        
        print(f"Found {len(valid_samples)} valid samples with all bands and labels")
        return valid_samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_band(self, band_path):
        """Load individual band with proper error handling"""
        try:
            if not os.path.exists(band_path):
                # print(f"File not found: {band_path}")
                return np.zeros(self.image_size, dtype=np.float32)
                
            if band_path.endswith(('.tif', '.tiff')):
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
            else:
                band_data = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE)
            
            if band_data is None:
                # print(f"Warning: Could not load {band_path}")
                return np.zeros(self.image_size, dtype=np.float32)
            
            # Resize if necessary
            if band_data.shape != self.image_size:
                band_data = cv2.resize(band_data, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            return band_data
        except Exception as e:
            # print(f"Error loading {band_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def __getitem__(self, idx):
        sample_files = self.samples[idx]
        # print(f"Loading sample {idx}: {sample_files}")  # Debug
        
        # Load all bands
        band_images = []
        for band in self.bands:
            if band in sample_files:
                band_path = os.path.join(self.root_dir, band, sample_files[band])
                band_data = self._load_band(band_path)
            else:
                band_data = np.zeros(self.image_size, dtype=np.float32)
            band_images.append(band_data)
        
        # Stack bands to create multi-channel image
        image = np.stack(band_images, axis=0)  # Shape: (n_bands, H, W)
        image = image.astype(np.float32)
        
        # Normalize each band
        for i in range(image.shape[0]):
            if np.max(image[i]) > 0 and np.max(image[i]) != np.min(image[i]):
                image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        
        # Load label
        label_path = os.path.join(self.root_dir, self.label_dir, sample_files['label'])
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
            # print(f"Error loading label {label_path}: {e}")
            label = np.zeros(self.image_size, dtype=np.float32)
        
        # Apply transformations
        if self.transform:
            # Convert to HWC format for albumentations (numpy array)
            image_hwc = image.transpose(1, 2, 0)  # This is numpy operation
            
            # Apply albumentations
            augmented = self.transform(image=image_hwc, mask=label)
            
            # albumentations with ToTensorV2 converts to tensors automatically
            image_tensor = augmented['image']  # Shape: (n_bands, H, W) - already tensor
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
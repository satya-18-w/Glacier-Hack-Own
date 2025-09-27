import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GlacierDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load multispectral image (5 channels)
        with rasterio.open(self.image_paths[idx]) as img:
            image = img.read()  # Shape: (5, H, W)
            image = image.astype(np.float32)
        
        # Load mask
        with rasterio.open(self.mask_paths[idx]) as mask:
            mask_data = mask.read(1)  # Shape: (H, W)
            mask_data = (mask_data > 0).astype(np.float32)  # Binary mask
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), 
                                       mask=mask_data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask_data)
            
        return image, mask.unsqueeze(0)

class DataLoaderFactory:
    def __init__(self, config):
        self.config = config
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        train_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0], 
                    self.config['data']['image_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                              rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),  # Adjust based on data
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0],
                    self.config['data']['image_size'][1]),
            A.Normalize(mean=[0.0]*5, std=[1.0]*5),
            ToTensorV2(),
        ])
        
        return {'train': train_transform, 'val': val_transform}
    
    def create_loaders(self, data_dir):
        # Assuming structure: data_dir/images/*.tif, data_dir/masks/*.tif
        image_paths = sorted([os.path.join(data_dir, 'images', f) 
                            for f in os.listdir(os.path.join(data_dir, 'images')) 
                            if f.endswith('.tif')])
        mask_paths = sorted([os.path.join(data_dir, 'masks', f) 
                           for f in os.listdir(os.path.join(data_dir, 'masks')) 
                           if f.endswith('.tif')])
        
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_paths, mask_paths, 
            test_size=self.config['data']['val_split'],
            random_state=42
        )
        
        train_dataset = GlacierDataset(train_img, train_mask, 
                                     self.transform['train'], is_train=True)
        val_dataset = GlacierDataset(val_img, val_mask, 
                                   self.transform['val'], is_train=False)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['data']['batch_size'],
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.config['data']['batch_size'],
                              shuffle=False, num_workers=4)
        
        return train_loader, val_loader
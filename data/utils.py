import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_dataset(dataset, train_size=0.8, random_state=42):
    """Split dataset into train and validation sets"""
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, train_size=train_size, random_state=random_state)
    return train_idx, val_idx

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    pixel_counts = {0: 0, 1: 0}
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        unique, counts = np.unique(label.numpy(), return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counts[int(u)] += c
    
    total_pixels = sum(pixel_counts.values())
    weights = {
        0: total_pixels / (2 * pixel_counts[0]),
        1: total_pixels / (2 * pixel_counts[1])
    }
    
    return weights
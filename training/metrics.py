import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(predictions, targets):
    """Calculate Matthews Correlation Coefficient"""
    pred_flat = np.array(predictions).flatten()
    target_flat = np.array(targets).flatten()
    
    # Ensure binary values
    pred_flat = (pred_flat > 0.5).astype(np.int32)
    target_flat = (target_flat > 0.5).astype(np.int32)
    
    return matthews_corrcoef(target_flat, pred_flat)

def calculate_iou(predictions, targets):
    """Calculate Intersection over Union"""
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(predictions, targets):
    """Calculate Dice Coefficient"""
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    return (2. * intersection + 1e-6) / (predictions.sum() + targets.sum() + 1e-6)
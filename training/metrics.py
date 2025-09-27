import numpy as np
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(predictions, targets):
    """Calculate Matthews Correlation Coefficient"""
    return matthews_corrcoef(targets.flatten(), predictions.flatten())

def calculate_iou(predictions, targets):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(predictions, targets).sum()
    union = np.logical_or(predictions, targets).sum()
    return intersection / union if union != 0 else 0

def calculate_precision_recall(predictions, targets):
    """Calculate precision and recall"""
    tp = np.logical_and(predictions == 1, targets == 1).sum()
    fp = np.logical_and(predictions == 1, targets == 0).sum()
    fn = np.logical_and(predictions == 0, targets == 1).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
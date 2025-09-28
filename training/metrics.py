import torch

def calculate_mcc(y_true, y_pred, threshold=0.5):
    """
    Calculate Matthews Correlation Coefficient from PyTorch tensors.

    Args:
        y_true (torch.Tensor): Ground truth masks (0 or 1).
        y_pred (torch.Tensor): Predicted masks (probabilities).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: MCC score.
    """
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

def calculate_iou(predictions, targets):
    """Calculate Intersection over Union"""
    intersection = (predictions & targets).sum()
    union = (predictions | targets).sum()
    return intersection / union if union != 0 else 0

def calculate_precision_recall(predictions, targets):
    """Calculate precision and recall"""
    tp = (predictions & targets).sum()
    fp = (predictions & ~targets).sum()
    fn = (~predictions & targets).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
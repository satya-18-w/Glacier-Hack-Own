import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_history(trainer, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(trainer.train_losses, label='Training Loss')
    ax1.plot(trainer.val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot MCC
    ax2.plot(trainer.val_mccs, label='Validation MCC', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MCC')
    ax2.legend()
    ax2.set_title('Validation Matthews Correlation Coefficient')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_prediction(image, mask, prediction, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # RGB composite (bands 2,3,4)
    rgb = image[[1, 2, 3], :, :].permute(1, 2, 0).numpy()
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Composite')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction.squeeze(), cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(rgb)
    axes[3].imshow(prediction.squeeze(), alpha=0.3, cmap='Reds')
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
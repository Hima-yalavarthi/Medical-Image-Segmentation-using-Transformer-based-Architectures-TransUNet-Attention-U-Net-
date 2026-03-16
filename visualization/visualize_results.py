import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_prediction(image, mask, prediction, save_path=None):
    """
    image: [C, H, W]
    mask: [1, H, W]
    prediction: [1, H, W] (after sigmoid and threshold)
    """
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Overlay Comparison")
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='green', alpha=0.3)
    plt.imshow(prediction, cmap='red', alpha=0.3)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_loss(train_losses, val_losses=None, save_path=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

import torch
import os
import matplotlib.pyplot as plt
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from utils.dataset_loader import MedicalDataset
from preprocessing.augmentation import get_val_transforms

def save_comparison():
    device = torch.device("cpu")
    img_size = 224
    
    # Load dataset
    dataset = MedicalDataset("dataset/images", "dataset/masks", transform=get_val_transforms(img_size))
    
    # Pick a sample
    idx = 10 # Sample index
    image, mask = dataset[idx]
    image_input = image.unsqueeze(0).to(device)
    
    # Load models
    model_att = AttentionUNet().to(device)
    model_att.load_state_dict(torch.load("checkpoints/attention_unet_best.pth", map_location=device))
    model_att.eval()
    
    model_trans = TransUNet().to(device)
    model_trans.load_state_dict(torch.load("checkpoints/transunet_best.pth", map_location=device))
    model_trans.eval()
    
    # Predict
    with torch.no_grad():
        pred_att = torch.sigmoid(model_att(image_input)).squeeze().cpu().numpy()
        pred_trans = torch.sigmoid(model_trans(image_input)).squeeze().cpu().numpy()
        
    # Plot
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Attention U-Net Pred")
    plt.imshow(pred_att > 0.5, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("TransUNet Pred")
    plt.imshow(pred_trans > 0.5, cmap='gray')
    plt.axis('off')
    
    os.makedirs("visualization/results", exist_ok=True)
    save_path = "visualization/results/comparison_sample.png"
    plt.savefig(save_path)
    print(f"Comparison saved to {save_path}")

if __name__ == "__main__":
    save_comparison()

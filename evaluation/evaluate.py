import torch
import os
import pandas as pd
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from utils.dataset_loader import get_dataloader
from evaluation.metrics import get_metrics
from preprocessing.augmentation import get_val_transforms
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_metrics = get_metrics(outputs, masks)
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
                
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return avg_metrics

def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on {device}")
    
    val_img_dir = "dataset/images" # Using same for demo
    val_mask_dir = "dataset/masks"
    val_loader = get_dataloader(val_img_dir, val_mask_dir, batch_size=4, shuffle=False, transform=get_val_transforms())
    
    results = []
    
    # 1. Attention U-Net
    model_att = AttentionUNet().to(device)
    att_checkpoint = "checkpoints/attention_unet_best.pth"
    if os.path.exists(att_checkpoint):
        model_att.load_state_dict(torch.load(att_checkpoint, map_location=device))
        print("Loaded Attention U-Net weights.")
    else:
        print(f"Warning: {att_checkpoint} not found. Using untrained weights.")
    
    print("Evaluating Attention U-Net...")
    metrics_att = evaluate_model(model_att, val_loader, device)
    metrics_att["Model"] = "Attention UNet"
    results.append(metrics_att)
    
    # 2. TransUNet
    model_trans = TransUNet().to(device)
    trans_checkpoint = "checkpoints/transunet_best.pth"
    if os.path.exists(trans_checkpoint):
        model_trans.load_state_dict(torch.load(trans_checkpoint, map_location=device))
        print("Loaded TransUNet weights.")
    else:
        print(f"Warning: {trans_checkpoint} not found. Using untrained weights.")
        
    print("Evaluating TransUNet...")
    metrics_trans = evaluate_model(model_trans, val_loader, device)
    metrics_trans["Model"] = "TransUNet"
    results.append(metrics_trans)
    
    df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df)
    df.to_csv("evaluation/model_comparison.csv", index=False)

if __name__ == "__main__":
    compare_models()

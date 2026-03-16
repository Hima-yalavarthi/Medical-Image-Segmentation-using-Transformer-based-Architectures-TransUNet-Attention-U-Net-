import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from training.loss import CombinedLoss
from utils.dataset_loader import get_dataloader
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from evaluation.metrics import get_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_metrics = get_metrics(outputs, masks)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
            
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            batch_metrics = get_metrics(outputs, masks)
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
                
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics

def main(model_name="attention_unet", epochs=10, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path setup (Assuming relative to project root)
    train_img_dir = "dataset/images"
    train_mask_dir = "dataset/masks"
    
    train_loader = get_dataloader(train_img_dir, train_mask_dir, batch_size=batch_size, transform=get_train_transforms())
    
    if model_name == "attention_unet":
        model = AttentionUNet().to(device)
    else:
        model = TransUNet().to(device)
        
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    best_dice = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        
        # Save checkpoint
        if train_metrics['dice'] > best_dice:
            best_dice = train_metrics['dice']
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
            print("Best model saved!")
            
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "transunet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    main(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

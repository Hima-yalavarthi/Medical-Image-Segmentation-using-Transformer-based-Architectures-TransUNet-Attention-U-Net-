import torch
import cv2
import numpy as np
import argparse
import os
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from preprocessing.augmentation import get_val_transforms

def predict(image_path, model_type, checkpoint_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    
    # Load model
    if model_type == "attention_unet":
        model = AttentionUNet().to(device)
    else:
        model = TransUNet().to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    orig_h, orig_w = image.shape
    transform = get_val_transforms(img_size)
    # Transform expects dict with 'image'
    aug = transform(image=image.astype(np.float32) / 255.0)
    image_tensor = aug['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()
    
    # Rescale mask back to original size
    pred_mask = (output > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Save result
    cv2.imwrite(output_path, pred_mask)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "transunet"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="prediction.png", help="Path to save predicted mask")
    args = parser.parse_args()
    
    predict(args.image, args.model, args.checkpoint, args.output)

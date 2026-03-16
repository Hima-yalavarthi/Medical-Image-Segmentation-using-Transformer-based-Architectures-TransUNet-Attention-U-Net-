import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

class MedicalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the directory with images.
            masks_dir (str): Path to the directory with masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(images_dir, "*")))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, "*")))
        
        # Simple validation
        if len(self.image_paths) != len(self.mask_paths):
            print(f"Warning: Number of images ({len(self.image_paths)}) doesn't match masks ({len(self.mask_paths)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image {img_path} or mask {mask_path}")

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure channel-first [1, H, W]
        if not isinstance(image, torch.Tensor):
            if image.ndim == 2:
                image = torch.from_numpy(image).unsqueeze(0)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
        
        if not isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = torch.from_numpy(mask).unsqueeze(0)
            else:
                mask = torch.from_numpy(mask).permute(2, 0, 1)

        # Force mask to be [1, H, W] specifically
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()
            
        return image, mask

def get_dataloader(images_dir, masks_dir, batch_size=8, shuffle=True, num_workers=4, transform=None):
    dataset = MedicalDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

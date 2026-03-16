import numpy as np
import cv2
import torch

def normalize_image(image):
    """
    Normalize image to range [0, 1].
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    return image

def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target size.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def extract_patches(image, mask, patch_size=(128, 128), stride=(64, 64)):
    """
    Extract patches from image and mask for high-resolution images.
    """
    img_h, img_w = image.shape[:2]
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    patches_img = []
    patches_mask = []
    
    for y in range(0, img_h - patch_h + 1, stride_h):
        for x in range(0, img_w - patch_w + 1, stride_w):
            patch_img = image[y:y+patch_h, x:x+patch_w]
            patch_mask = mask[y:y+patch_h, x:x+patch_w]
            patches_img.append(patch_img)
            patches_mask.append(patch_mask)
            
    return np.array(patches_img), np.array(patches_mask)

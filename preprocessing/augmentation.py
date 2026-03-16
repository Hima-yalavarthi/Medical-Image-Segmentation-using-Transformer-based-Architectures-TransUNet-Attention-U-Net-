import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transforms(img_size=224):
    """
    Returns training transformations including augmentations.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    """
    Returns validation transformations (only resizing and normalization).
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

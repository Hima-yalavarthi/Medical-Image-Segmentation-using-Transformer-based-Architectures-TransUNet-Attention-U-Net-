import os
import pandas as pd
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from glob import glob

def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_metadata(path):
    # slice_0001_266_266_1.50_1.50.png
    parts = os.path.basename(path).split('_')
    return int(parts[1]), int(parts[2]), int(parts[3])

def prepare_data(csv_path, train_dir, output_dir):
    df = pd.read_csv(csv_path)
    
    # Filter rows with segmentation
    df = df[df['segmentation'].notna()]
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # Group by id to combine multiple segments (stomach, large_bowel, small_bowel)
    grouped = df.groupby('id')
    
    for id_name, group in tqdm(grouped, desc="Processing masks"):
        # ID format: case123_day20_slice_0001
        parts = id_name.split('_')
        case_id = parts[0]
        day_id = parts[1]
        slice_id = parts[2] + "_" + parts[3]
        
        # Find the source image path
        search_pattern = os.path.join(train_dir, case_id, f"{case_id}_{day_id}", "scans", f"{slice_id}*")
        src_paths = glob(search_pattern)
        
        if not src_paths:
            continue
            
        src_path = src_paths[0]
        slice_num, h, w = get_metadata(src_path)
        
        # Combine masks for all classes in this slice
        full_mask = np.zeros((h, w), dtype=np.uint8)
        for _, row in group.iterrows():
            mask = rle_decode(row['segmentation'], (h, w))
            full_mask = np.maximum(full_mask, mask)
        
        # Save image and mask
        dst_name = f"{id_name}.png"
        shutil.copy(src_path, os.path.join(output_dir, "images", dst_name))
        cv2.imwrite(os.path.join(output_dir, "masks", dst_name), full_mask * 255)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train/ folder")
    parser.add_argument("--output", type=str, default="dataset", help="Target dataset folder")
    args = parser.parse_args()
    
    prepare_data(args.csv, args.train_dir, args.output)

import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, filename):
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading Kvasir-SEG")
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def setup_kvasir():
    # URL for Kvasir-SEG (Polyp Segmentation) - ~46MB
    url = "https://datasets.simula.no/kvasir-seg/kvasir-seg.zip"
    zip_name = "kvasir-seg.zip"
    
    print("Starting Kvasir-SEG dataset setup...")
    
    # Clean up previous failed attempts
    if os.path.exists(zip_name):
        os.remove(zip_name)
    
    # 1. Download
    download_file(url, zip_name)
    
    # 2. Extract
    print("Extracting files...")
    if os.path.exists("temp_kvasir"):
        shutil.rmtree("temp_kvasir")
        
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall("temp_kvasir")
    
    # 3. Organize into project structure
    print("Organizing dataset...")
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/masks", exist_ok=True)
    
    # The zip might extract to 'Kvasir-SEG' or 'kvasir-seg' or directly
    extracted_dirs = [d for d in os.listdir("temp_kvasir") if os.path.isdir(os.path.join("temp_kvasir", d))]
    
    if extracted_dirs:
        root_extracted = os.path.join("temp_kvasir", extracted_dirs[0])
    else:
        root_extracted = "temp_kvasir"
        
    temp_img_dir = os.path.join(root_extracted, "images")
    temp_mask_dir = os.path.join(root_extracted, "masks")
    
    if not os.path.exists(temp_img_dir):
        print(f"Error: Could not find images directory at {temp_img_dir}")
        return

    # Move images
    for img_name in os.listdir(temp_img_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(os.path.join(temp_img_dir, img_name), os.path.join("dataset/images", img_name))
        
    # Move masks
    for mask_name in os.listdir(temp_mask_dir):
        if mask_name.endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(os.path.join(temp_mask_dir, mask_name), os.path.join("dataset/masks", mask_name))
        
    # 4. Cleanup
    print("Cleaning up...")
    shutil.rmtree("temp_kvasir")
    os.remove(zip_name)
    
    print("\nSuccess! Kvasir-SEG dataset is ready in 'dataset/' folder.")
    print(f"Total images: {len(os.listdir('dataset/images'))}")
    print(f"Total masks: {len(os.listdir('dataset/masks'))}")
    print("\nYou can now start training using:")
    print("python training/train.py --model transunet --epochs 10")

if __name__ == "__main__":
    setup_kvasir()

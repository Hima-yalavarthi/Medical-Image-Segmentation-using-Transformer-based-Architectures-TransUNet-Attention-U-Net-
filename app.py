import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from preprocessing.augmentation import get_val_transforms

# Page config
st.set_page_config(page_title="Medical Image Segmentation", layout="wide")

st.title("🩺 Medical Image Segmentation App")
st.markdown("""
Upload a medical scan (endoscopy, MRI, etc.) to generate precise segmentation masks using state-of-the-art AI models.
""")

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose Segmentation Model",
    ["Attention U-Net", "TransUNet"]
)

model_map = {
    "Attention U-Net": ("attention_unet", "checkpoints/attention_unet_best.pth"),
    "TransUNet": ("transunet", "checkpoints/transunet_best.pth")
}

model_type, checkpoint_path = model_map[model_choice]

@st.cache_resource
def load_model(m_type, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if m_type == "attention_unet":
        model = AttentionUNet().to(device)
    else:
        model = TransUNet().to(device)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model, device
    return None, device

model, device = load_model(model_type, checkpoint_path)

if model is None:
    st.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload a Medical Scan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Scan")
            st.image(opencv_image, channels="GRAY", use_container_width=True)
            
        if st.button("Run Segmentation"):
            with st.spinner(f"Running {model_choice} inference..."):
                # Preprocess
                img_size = 224
                orig_h, orig_w = opencv_image.shape
                transform = get_val_transforms(img_size)
                aug = transform(image=opencv_image.astype(np.float32) / 255.0)
                image_tensor = aug['image'].unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()
                
                # Rescale back
                pred_mask = (output > 0.5).astype(np.uint8) * 255
                pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # Create overlay
                color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                color_mask[pred_mask_resized > 0] = [255, 0, 0] # Red mask
                
                # Convert original to BGR for overlay
                orig_bgr = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(orig_bgr, 0.7, color_mask, 0.3, 0)
                
                with col2:
                    st.subheader("Segmentation Result")
                    st.image(pred_mask_resized, caption="Binary Mask", use_container_width=True)
                
                st.divider()
                st.subheader("Precise Overlay Visualization")
                st.image(overlay, caption="Red area indicates segmented region", use_container_width=True)
                
                st.success("Analysis complete!")
                
                # Display metrics placeholder or actuals
                st.sidebar.info(f"Currently using {model_choice} optimized for Kvasir-SEG.")


from PIL import Image

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch.nn as nn
import numpy as np

from src.model_utils import load_model, load_and_transform_image, prepare_image, get_device 

#---------------------------------------------------------------------------------------------------------------------------------
# Attention Map Functions

def get_attention_maps(model, img, version, device=None):
    """Get attention maps from the model."""
    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    return attentions, nh

def process_attention_maps(attentions, nh, patch_size, dim_feature_map, version):
    """Process the attention maps.""" 
    if version == "dinov1":
        attentions = attentions.reshape(nh, dim_feature_map[0]//2, dim_feature_map[1]//2)
    elif version in ["dinov2_small", "dinov2_large"]:
        attentions = attentions[:, 4:]  # Specific to DINOv2 
        attentions = attentions.reshape(nh, dim_feature_map[0], dim_feature_map[1])
    else:
        raise ValueError("Unsupported model version! Choose 'dinov1', 'dinov2_small', or 'dinov2_large'.")
    
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions_mean = np.mean(attentions, axis=0)
    return attentions, attentions_mean

#---------------------------------------------------------------------------------------------------------------------------------
# Plotting Functions

def plot_attentions(img0, attentions, attentions_mean, overlay=False, dim_factor=1):
    """Plot the results with optional heatmap overlay."""
    img0 = img0.resize((img0.width // dim_factor, img0.height // dim_factor), Image.BILINEAR)
    plt.figure(figsize=(6, 6), dpi=200)

    plt.subplot(3, 3, 1)
    plt.title("Original", size=6)
    plt.imshow(img0)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Attentions Mean", size=6)
    if overlay:
        plt.imshow(img0)
        plt.imshow(attentions_mean, cmap='jet', alpha=0.5)  # Overlay with heatmap
    else:
        plt.imshow(attentions_mean, cmap='jet')
    plt.axis("off")

    for i in range(min(len(attentions), 6)):
        plt.subplot(3, 3, i + 4)
        plt.title(f"Attentions {i}", size=6)
        if overlay:
            plt.imshow(img0)
            plt.imshow(attentions[i], cmap='jet', alpha=0.5)  # Overlay with heatmap
        else:
            plt.imshow(attentions[i], cmap='jet')
        plt.axis("off")

    plt.show()


def plot_attentions_smooth(img0, attentions, attentions_mean, overlay=False, dim_factor=1, blur_radius=5):
    """Plot the results with attention blobs and Gaussian blur for smoother visualization."""
    
    # Resize the original image to match the attention map dimensions
    img_resized = img0.resize(
        (img0.width // dim_factor, img0.height // dim_factor), Image.BILINEAR
    )
    
    # Apply Gaussian blur to the attention maps
    def create_blurred_attention(attention, blur_radius):
        attention = (attention - attention.min()) / (attention.max() - attention.min())  # Normalize attention map
        attention_blurred = gaussian_filter(attention, sigma=blur_radius)  # Apply Gaussian blur
        return attention_blurred
    
    attentions_blurred = [create_blurred_attention(att, blur_radius) for att in attentions]
    attentions_mean_blurred = create_blurred_attention(attentions_mean, blur_radius)
    
    plt.figure(figsize=(6, 6), dpi=200)

    # Plot the original image
    plt.subplot(3, 3, 1)
    plt.title("Original", size=6)
    plt.imshow(img0)
    plt.axis("off")

    # Plot the mean attention map with blobs and blur
    plt.subplot(3, 3, 2)
    plt.title("Attentions Mean (Blurred)", size=6)
    if overlay:
        plt.imshow(img_resized)  # Show resized image
        plt.imshow(attentions_mean_blurred, cmap='jet', alpha=0.5)  # Overlay blurred attention blobs
    else:
        plt.imshow(attentions_mean_blurred, cmap='jet')  # Show blurred attention blobs only
    plt.axis("off")

    # Plot individual blurred attention heads
    for i in range(min(len(attentions_blurred), 6)):
        plt.subplot(3, 3, i + 4)
        plt.title(f"Attention {i + 1} (Blurred)", size=6)
        if overlay:
            plt.imshow(img_resized)  # Show resized image
            plt.imshow(attentions_blurred[i], cmap='jet', alpha=0.5)  # Overlay blurred attention blobs
        else:
            plt.imshow(attentions_blurred[i], cmap='jet')  # Show blurred attention blobs only
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------------
# Display Attention Maps from specified model on image
    
def display_attention(image_path, version="dinov1", overlay=False):
    # Get device
    device = get_device()

    # Load model and image
    model, patch_size    = load_model(version=version, device=device, inference=True)
    img                  = load_and_transform_image(image_path=image_path, add_batch_dim=False)
    img, dim_feature_map = prepare_image(image=img, patch_size=patch_size)

    # Get attention maps
    attentions, nh = get_attention_maps(model, img, version, device)

    # Process attention maps
    dim_factor = 2 if version == "dinov1" else 1  # Adjust for dinov1
    attentions, attentions_mean = process_attention_maps(attentions, nh, patch_size, dim_feature_map, version)

    # Plot results with or without overlay
    image = Image.open(image_path).convert('RGB')
    plot_attentions_smooth(image, attentions, attentions_mean, overlay=overlay, dim_factor=dim_factor)

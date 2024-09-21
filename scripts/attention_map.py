import torch
import torchvision.transforms as pth_transforms

from PIL import Image

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import numpy as np

from dinov2.models.vision_transformer import vit_small, vit_large

# Constants
PATCH_SIZE_V1 = 8
PATCH_SIZE_V2 = 14
IMAGE_SIZE    = 526  # Used for DINOv2

#----------------------------------------------------------------------------------------------------------------------------------
# Load

def load_model(version="dinov1", device=None):
    """Load DINOv1 or DINOv2 model."""

    print(f"Loading model {version}...")
    
    if version == "dinov1":
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        
    elif version == "dinov2_small":
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg') 
        model = vit_small(patch_size=PATCH_SIZE_V2, img_size=IMAGE_SIZE, init_values=1.0, block_chunks=0, num_register_tokens=pretrained_model.num_register_tokens)
        model.load_state_dict(pretrained_model.state_dict())
        
    elif version == "dinov2_large":
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        model = vit_large(patch_size=PATCH_SIZE_V2, img_size=IMAGE_SIZE, init_values=1.0, block_chunks=0, num_register_tokens=pretrained_model.num_register_tokens)
        model.load_state_dict(pretrained_model.state_dict()) 
    else:
        raise ValueError("Unsupported model version! Choose 'dinov1', 'dinov2_small', or 'dinov2_large'.")

    model.to(device)
   
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model

def load_and_transform_image(image_path, version):
    """Load and transform the image."""
    img = Image.open(image_path).convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    return img

#----------------------------------------------------------------------------------------------------------------------------------
# Data processing

def prepare_image(img, patch_size):
    """Prepare the image for the model."""
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size 
    return img, (w_featmap, h_featmap)

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


#----------------------------------------------------------------------------------------------------------------------------------
# Plots

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
    
#----------------------------------------------------------------------------------------------------------------------------------
# Entry Point
    
def display_attention(image_path, version="dinov1", overlay=False):
    """Main function to display attention maps with an optional overlay on the input image."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set the appropriate patch size for the model version
    if version == "dinov1":
        patch_size = PATCH_SIZE_V1
    elif version in ["dinov2_small", "dinov2_large"]:
        patch_size = PATCH_SIZE_V2
    else:
        raise ValueError("Unsupported model version! Choose 'dinov1', 'dinov2_small', or 'dinov2_large'.")

    # Load model and image
    model = load_model(version=version, device=device)
    img = load_and_transform_image(image_path, version)
    img, dim_feature_map = prepare_image(img, patch_size)

    # Get attention maps
    attentions, nh = get_attention_maps(model, img, version, device)

    # Process attention maps
    dim_factor = 2 if version == "dinov1" else 1  # Adjust for dinov1
    attentions, attentions_mean = process_attention_maps(attentions, nh, patch_size, dim_feature_map, version)

    # Plot results with or without overlay
    plot_attentions_smooth(Image.open(image_path).convert('RGB'), attentions, attentions_mean, overlay=overlay, dim_factor=dim_factor)

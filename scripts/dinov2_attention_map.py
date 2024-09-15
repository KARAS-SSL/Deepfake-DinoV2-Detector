import os
import torch
import torch.nn as nn
import torchvision.transforms as pth_transforms
from PIL import Image
import matplotlib.pyplot as plt
from dinov2.models.vision_transformer import vit_small, vit_large

import numpy as np

PATCH_SIZE = 14
IMAGE_SIZE = (952, 952)
OUTPUT_DIR = '.'

def load_model(device, large=False):

    pretrained_model = None
    model = None
    if not large:
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        model = vit_small(
            patch_size=PATCH_SIZE,
            img_size=526,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=pretrained_model.num_register_tokens 
        )
    else:
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        model = vit_large(
            patch_size=PATCH_SIZE,
            img_size=526,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=pretrained_model.num_register_tokens 
        ) 
        
    model.load_state_dict(pretrained_model.state_dict()) 
    
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model

def load_and_transform_image(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    transform = pth_transforms.Compose([
        # pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    return img

def prepare_image(img, patch_size):
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    return img, (w_featmap, h_featmap)

def get_attention_maps(model, img, device):
    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # attentions[:, 283] = 0  # removing a suspicious high-attention pixel
    return attentions, nh

def process_attention_maps(attentions, nh, patch_size, dim_feature_map):
    attentions = attentions[:, 4:]
    attentions = attentions.reshape(nh, dim_feature_map[0], dim_feature_map[1])
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions_mean = np.mean(attentions, axis=0)    
    return attentions, attentions_mean

def plot_results(img0, attentions, attentions_mean):
    plt.figure(figsize=(6,6), dpi=200)
    
    plt.subplot(3, 3, 1)
    plt.title("Original", size=6)
    plt.imshow(img0)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Attentions Mean", size=6)
    plt.imshow(attentions_mean)
    plt.axis("off")
    
    for i in range(min(len(attentions), 6)):
        plt.subplot(3, 3, i+4)
        plt.title("Attentions "+str(i), size=6)
        plt.imshow(attentions[i])
        plt.axis("off")
    
    plt.show()


def display_attention(image_path, large=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = load_model(device, large)
    img = load_and_transform_image(image_path, IMAGE_SIZE)
    img, dim_feature_map = prepare_image(img, PATCH_SIZE)

    attentions, nh = get_attention_maps(model, img, device)   
    attentions, attentions_mean = process_attention_maps(attentions, nh, PATCH_SIZE, dim_feature_map) 
    plot_results(Image.open(image_path).convert('RGB'), attentions, attentions_mean)
    

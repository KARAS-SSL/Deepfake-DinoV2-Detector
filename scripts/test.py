import torch
import torchvision.transforms as pth_transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

PATCH_SIZE = 8

def load_model():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def load_and_transform_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = pth_transforms.Compose([
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

def get_attention_maps(model, img):
    attentions = model.get_last_selfattention(img)
    nh = attentions.shape[1]  # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    return attentions, nh

def process_attention_maps(attentions, nh, patch_size, dim_feature_map, threshold=0.6):
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    
    th_attn = th_attn.reshape(nh, dim_feature_map[0]//2, dim_feature_map[1]//2).float() 
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
    attentions = attentions.reshape(nh, dim_feature_map[0]//2, dim_feature_map[1]//2)
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

def main(image_path):
    model = load_model()
    img   = load_and_transform_image(image_path)
    img, dim_feature_map = prepare_image(img, PATCH_SIZE)
    
    attentions, nh = get_attention_maps(model, img)
    attentions, attentions_mean = process_attention_maps(attentions, nh, PATCH_SIZE, dim_feature_map)
    
    plot_results(Image.open(image_path).convert('RGB'), attentions, attentions_mean)

if __name__ == "__main__":
    main("your_image.jpg")


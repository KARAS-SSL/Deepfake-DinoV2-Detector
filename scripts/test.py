import torch
import torchvision.transforms as pth_transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import torch.nn as nn

import numpy as np


patch_size = 8

# Load the DINOv2 model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for p in model.parameters():
    p.requires_grad = False
        
model.eval()
# model.to(device)

# Load and transform an image
img0 = Image.open("your_image.jpg")
img0 = img0.convert('RGB')

# Get Attention Maps
transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
img = transform(img0)

# make the image divisible by the patch size
w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
img = img[:, :w, :h].unsqueeze(0)

w_featmap = img.shape[-2] // patch_size
h_featmap = img.shape[-1] // patch_size

attentions = model.get_last_selfattention(img)   #img.cuda()

nh = attentions.shape[1] # number of head

# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

# we keep only a certain percentage of the mass
val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)
cumval = torch.cumsum(val, dim=1)

threshold = 0.6 # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
th_attn = cumval > (1 - threshold)
idx2 = torch.argsort(idx)
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]
    
th_attn = th_attn.reshape(nh, w_featmap//2, h_featmap//2).float()

# interpolate
th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
attentions_mean = np.mean(attentions, axis=0)

print(attentions.shape)


plt.figure(figsize=(6,6), dpi=200)

plt.subplot(3, 3, 1)
plt.title("Original",size=6)
plt.imshow(img0)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Attentions Mean",size=6)
plt.imshow(attentions_mean)
plt.axis("off")

for i in range(6):
    plt.subplot(3, 3, i+4)
    plt.title("Attentions "+str(i),size=6)
    plt.imshow(attentions[i])
    plt.axis("off")

plt.show()
# plt.figure(figsize=(10, 10))

# Original image
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis('off')

# # Attention map
# plt.subplot(1, 2, 2)
# plt.imshow(image)
# plt.imshow(attention_map, cmap='jet', alpha=0.5)
# plt.title("Attention Map")
# plt.axis('off')

# plt.show()


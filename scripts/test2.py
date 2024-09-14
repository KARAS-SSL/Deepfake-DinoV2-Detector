import torch
from transformers import Dinov2Model, AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained DinoV2 model and image processor using AutoImageProcessor
model = Dinov2Model.from_pretrained('facebook/dinov2-base')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

# Load and preprocess the input image
image_path = "your_image.jpg"  # Replace with your image path
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")

# Forward pass to get outputs including attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Access the attentions (list of tensor for each layer)
attentions = outputs.attentions  # (layers, batch_size, num_heads, sequence_length, sequence_length)

# Set up the grid layout: define rows and columns for visualization
num_layers = len(attentions)
grid_size = int(np.ceil(np.sqrt(num_layers)))  # Create a square grid that fits all layers

# Create a matplotlib figure
plt.figure(figsize=(12, 12))

# Iterate through each layer's attentions
for i, layer_attention in enumerate(attentions):
    # Average attention across all heads
    avg_attention = layer_attention.mean(dim=1)  # (batch_size, sequence_length, sequence_length)

    # Get the attention map for the [CLS] token
    sequence_length = avg_attention.shape[-1]
    patch_size = 16

    # Compute the number of patches (sequence_length - 1 because of the [CLS] token)
    num_patches = sequence_length - 1
    h_patches = w_patches = int(num_patches ** 0.5)  # Assuming a square grid of patches

    # Reshape the attention to the grid of patches
    cls_attention = avg_attention[0, 0, 1:].reshape(h_patches, w_patches).cpu().numpy()

    # Resize the attention map to the size of the original image
    cls_attention_resized = np.array(Image.fromarray(cls_attention).resize(image.size, Image.BILINEAR))

    # Normalize for visualization
    cls_attention_resized = (cls_attention_resized - cls_attention_resized.min()) / (cls_attention_resized.max() - cls_attention_resized.min())

    # Plot the attention map in the grid
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(image)
    plt.imshow(cls_attention_resized, cmap='jet', alpha=0.5)  # Overlay attention map
    plt.axis('off')
    plt.title(f"Layer {i + 1}")

# Adjust layout and display the grid
plt.tight_layout()
plt.show()


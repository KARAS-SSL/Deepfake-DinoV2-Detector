import torch
from transformers import Dinov2Model, AutoImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Function to preprocess the image
def preprocess_image(image_path, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    return image, inputs


# Function to get attention maps from the model
def get_attention_maps(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List of attention maps (layers, batch_size, num_heads, seq_len, seq_len)
    return attentions


# Function to visualize attention maps for each layer
def display_attention_layers(image, attentions):
    num_layers = len(attentions)
    grid_size = int(np.ceil(np.sqrt(num_layers)))  # Define grid size for subplots

    plt.figure(figsize=(12, 12))
    for i, layer_attention in enumerate(attentions):
        avg_attention = layer_attention.mean(dim=1)  # Average across heads
        sequence_length = avg_attention.shape[-1]
        num_patches = sequence_length - 1
        h_patches = w_patches = int(num_patches ** 0.5)  # Assuming square grid of patches

        cls_attention = avg_attention[0, 0, 1:].reshape(h_patches, w_patches).cpu().numpy()
        cls_attention_resized = np.array(Image.fromarray(cls_attention).resize(image.size, Image.BILINEAR))
        cls_attention_resized = (cls_attention_resized - cls_attention_resized.min()) / (
                    cls_attention_resized.max() - cls_attention_resized.min())

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(image)
        plt.imshow(cls_attention_resized, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(f"Layer {i + 1}")

    plt.tight_layout()
    plt.show()


# Function to compute and display the average attention map across all layers
def display_average_attention(image, attentions):
    num_layers = len(attentions)
    average_attention_across_layers = None

    for layer_attention in attentions:
        avg_attention = layer_attention.mean(dim=1)  # Average across heads
        if average_attention_across_layers is None:
            average_attention_across_layers = avg_attention
        else:
            average_attention_across_layers += avg_attention

    # Average across all layers
    average_attention_across_layers /= num_layers

    sequence_length = average_attention_across_layers.shape[-1]
    num_patches = sequence_length - 1
    h_patches = w_patches = int(num_patches ** 0.5)

    cls_attention_avg = average_attention_across_layers[0, 0, 1:].reshape(h_patches, w_patches).cpu().numpy()
    cls_attention_resized = np.array(Image.fromarray(cls_attention_avg).resize(image.size, Image.BILINEAR))
    cls_attention_resized = (cls_attention_resized - cls_attention_resized.min()) / (
                cls_attention_resized.max() - cls_attention_resized.min())

    # Visualize original image and averaged attention map
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(cls_attention_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title("Average Attention Across Layers")

    plt.show()


# Main function to process the image, get attention maps, and display both individual layers and the average map
def main(image_path):
    model = Dinov2Model.from_pretrained('facebook/dinov2-base')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    # Preprocess image
    image, inputs = preprocess_image(image_path, processor)

    # Get attention maps from the model
    attentions = get_attention_maps(model, inputs)

    # Display attention layers
    display_attention_layers(image, attentions)

    # Display average attention across layers
    display_average_attention(image, attentions)


# Run the main function with your image path
image_path = "your_image.jpg"  # Replace with the path to your image
main(image_path)



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap.umap_ as umap

from src.model_utils import load_model, load_and_transform_image, get_device

#----------------------------------------------------------------------------------------------------------------------------------
# Embedding Creation Functions

IMGSZ_PATCH_RATIO = 16

def create_image_embedding(model, patch_size, image_path, device):
    """Load an image, process it, and return its embedding."""
    img = load_and_transform_image(image_path=image_path, image_size=patch_size*IMGSZ_PATCH_RATIO, add_batch_dim=True)
    with torch.no_grad(): 
        embedding = model(img.to(device))
    return embedding.cpu().numpy()

def save_embedding(embedding, output_dir, image_filename):
    """Save the embedding as a .npy file."""
    embedding_filename = os.path.splitext(image_filename)[0] + ".npy"
    embedding_path = os.path.join(output_dir, embedding_filename)
    np.save(embedding_path, embedding)

def create_embeddings(input_dir, output_dir, version="dinov1", max_images_number=None):
    """Create embeddings for the images in the input directory and save them to the output directory."""
    device = get_device()

    # Load model
    model, patch_size = load_model(device=device, version=version, inference=True)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Get the list of image files
    image_filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    image_filenames = sorted(image_filenames)

    # Optionally limit the number of images
    if max_images_number is not None:
        image_filenames = image_filenames[0:max_images_number]

    # Process each image file in the input directory with a progress bar
    for i in tqdm(range(0, len(image_filenames)), desc="Processing images"): 
        image_path = os.path.join(input_dir, image_filenames[i])

        # Create the embedding and save it
        embedding = create_image_embedding(model, patch_size, image_path, device)
        save_embedding(embedding, output_dir, image_filenames[i])

#----------------------------------------------------------------------------------------------------------------------------------
# Embedding Display Functions

def load_embeddings(input_dir_fake, input_dir_real):
    """Load embeddings from the specified fake and real directories."""
    embeddings = []
    labels     = []
    
    for folder, label in zip([input_dir_fake, input_dir_real], ['fake', 'real']):
        for filename in tqdm(os.listdir(folder), desc=f"Loading embeddings from {folder}"):
            if filename.endswith(".npy"):
                embedding = np.load(os.path.join(folder, filename))
                embeddings.append(embedding)
                labels.append(label)  # Label all embeddings based on the folder

    return np.vstack(embeddings), labels

def reduce_dimensions(embeddings, algorithm='tsne'):
    """Apply t-SNE or UMAP to reduce the dimensions of the embeddings based on algorithm."""
    if algorithm == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, verbose=1)
        return reducer.fit_transform(embeddings)
    elif algorithm == 'umap':
        reducer = umap.UMAP(verbose=True)
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError("Invalid algorithm. Choose either 'tsne' or 'umap'.")

def plot_embeddings(reduced_result, labels, algorithm='tsne'):
    """Plot the reduced embeddings with colors corresponding to the labels."""
    print("Plotting the embeddings...")
    plt.figure(figsize=(8, 8))

    unique_labels = set(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(reduced_result[mask, 0], reduced_result[mask, 1], color=colors(i), alpha=0.5, label=label)

    plt.title('Embeddings Visualization -- ' + algorithm.upper())
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title="Classes")
    plt.tight_layout()
    plt.show()

def display_embeddings(input_dir_fake, input_dir_real, algorithm='tsne'):
    """Main function to load embeddings, reduce dimensions, and plot them."""
    embeddings, labels = load_embeddings(input_dir_fake, input_dir_real)
    reduced_result     = reduce_dimensions(embeddings, algorithm=algorithm)
    plot_embeddings(reduced_result, labels, algorithm=algorithm) 


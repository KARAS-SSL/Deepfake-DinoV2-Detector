
import os     # Use to interface with the terminal 
from fabric import task

## Scripts Imports
from src.visualization.attention_map import display_attention
from src.visualization.image_embedding import create_embeddings, display_embeddings

@task
def DisplayAttention(c): 
    img_path = "datasets/Xhlulu/real-vs-fake/train/real/00015.jpg"
    display_attention(img_path, version="dinov1", overlay=True)
    display_attention(img_path, version="dinov2_small", overlay=True)
    display_attention(img_path, version="dinov2_large", overlay=True)

@task
def CreateEmbeddings(c): 
    # train/fake
    input_dir  = "datasets/Xhlulu/real-vs-fake/train/fake"
    output_dir = "embeddings/Xhlulu/train/fake"
    create_embeddings(input_dir, output_dir, version="dinov1", max_images_number=10000)

    # train/real
    input_dir  = "datasets/Xhlulu/real-vs-fake/train/real"
    output_dir = "embeddings/Xhlulu/train/real"
    create_embeddings(input_dir, output_dir, version="dinov1", max_images_number=10000) 
    
@task 
def DisplayEmbeddings(c):
    input_dir_fake = "embeddings/Xhlulu/train/fake"
    input_dir_real = "embeddings/Xhlulu/train/real"
    # display_embeddings(input_dir_fake, input_dir_real, algorithm="tsne")
    display_embeddings(input_dir_fake, input_dir_real, algorithm="umap") 


"""
TODO LIST 

- Pretext Task Code, finetunning 
- Downstram Training and Testing 
- Baseline with CNNs 
- Code to Plot 
- Code to Evaluate Classification

"""

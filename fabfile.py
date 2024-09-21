
import os     # Use to interface with the terminal 
from fabric import task

## Scripts Imports
from src.visualization.attention_map import display_attention

@task
def DisplayAttention(c): 
    img_path = "datasets/Xhlulu/real-vs-fake/train/real/00015.jpg"
    display_attention(img_path, version="dinov1", overlay=True)
    display_attention(img_path, version="dinov2_small", overlay=True)
    display_attention(img_path, version="dinov2_large", overlay=True)

@task
def CreateEmbeddings(c): 
    input_dir  = "datasets/Xhlulu/real-vs-fake/train/"
    output_dir = "embeddings/Xhlulu/"
    create_embeddings(input_dir, output_dir)

@task 
def DisplayEmbeddings(c):
    input_dir  = "embeddings/"
    display_embeddings(img_path)

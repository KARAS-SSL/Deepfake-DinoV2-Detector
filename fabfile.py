
import os     # Use to interface with the terminal 
from fabric import task

## Scripts Imports
from scripts.attention_map import display_attention

@task
def map(c): 
    img_path = "datasets/Xhlulu/real-vs-fake/train/real/00015.jpg"
    display_attention(img_path, version="dinov1", overlay=True)
    display_attention(img_path, version="dinov2_small", overlay=True)
    display_attention(img_path, version="dinov2_large", overlay=True) 

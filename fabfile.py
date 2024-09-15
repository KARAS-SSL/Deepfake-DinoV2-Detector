
import os     # Use to interface with the terminal 
from fabric import task

## Scripts Imports
from scripts.test import display_attention as dinov1_display_attention
from scripts.test2 import display_attention as dinov2_display_attention
##

@task
def map(c): 
    img_path = "datasets/Xhlulu/real-vs-fake/train/real/00015.jpg"
    dinov1_display_attention(img_path) 
    dinov2_display_attention(img_path, large=False) 
    dinov2_display_attention(img_path, large=True)
    print("Download Datasets...")

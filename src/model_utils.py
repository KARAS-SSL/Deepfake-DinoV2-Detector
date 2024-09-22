
from PIL import Image

import torch
import torchvision.transforms as pth_transforms
from dinov2.models.vision_transformer import vit_small, vit_large

# Constants
PATCH_SIZE_V1 = 8
PATCH_SIZE_V2 = 14
IMAGE_SIZE    = 526  # Used for DINOv2

#----------------------------------------------------------------------------------------------------------------------------------
# Loading Model and Preprocessing Images

def load_model(version="dinov1", device=None, inference=False):
    """Load DINOv1 or DINOv2 model."""
    print(f"Loading model {version}...")
   
    patch_size = None

    # Load model
    if version == "dinov1":
        patch_size = PATCH_SIZE_V1
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') 
    elif version == "dinov2_small":
        patch_size       = PATCH_SIZE_V2 
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg') 
        model = vit_small(patch_size=PATCH_SIZE_V2, img_size=IMAGE_SIZE, init_values=1.0, block_chunks=0, num_register_tokens=pretrained_model.num_register_tokens)
        model.load_state_dict(pretrained_model.state_dict())
    elif version == "dinov2_large":
        patch_size       = PATCH_SIZE_V2 
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        model = vit_large(patch_size=PATCH_SIZE_V2, img_size=IMAGE_SIZE, init_values=1.0, block_chunks=0, num_register_tokens=pretrained_model.num_register_tokens)
        model.load_state_dict(pretrained_model.state_dict()) 
    else:
        raise ValueError("Unsupported model version! Choose 'dinov1', 'dinov2_small', or 'dinov2_large'.")

    # Move model to device
    model.to(device) 
       
    # Set model to inference mode
    # 1. Freeze model parameters
    # 2. Set model to evaluation mode, disable dropout and batch normalization
    if inference:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        
    return model, patch_size

def load_and_transform_image(image_path, image_size=None, add_batch_dim=False):
    """Load and transform the image."""
    img = Image.open(image_path).convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(), 
        pth_transforms.Resize((image_size, image_size), antialias=True) if image_size else pth_transforms.Lambda(lambda x: x),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img = transform(img) 
    if add_batch_dim: img = img.unsqueeze(0) 
    return img

#----------------------------------------------------------------------------------------------------------------------------------
# Helper Functions

def prepare_image(image, patch_size):
    """Prepare the image for the model."""
    w, h = image.shape[1] - image.shape[1] % patch_size, image.shape[2] - image.shape[2] % patch_size
    image = image[:, :w, :h].unsqueeze(0)
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size 
    return image, (w_featmap, h_featmap)

def get_device():
    """Check for CUDA availability and set device."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 

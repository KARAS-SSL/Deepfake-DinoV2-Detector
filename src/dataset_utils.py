
import os
import torchvision.transforms as pth_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Function to create the output folder and subfolder for saving the model and results
def create_output_subfolder(output_folder):
    # Get the number of subfolders in output_folder
    run_id = len(next(os.walk(output_folder))[1]) + 1
    run_folder = os.path.join(output_folder, f'run{run_id}')
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

# Function to load the dataset
def load_dataset(train_path, val_path, image_size, batch_size):
    # Transforms applied to all images
    transform = pth_transforms.Compose([
        pth_transforms.Resize((image_size, image_size), antialias=True),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load training and validation dataset
    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset   = ImageFolder(root=val_path, transform=transform)

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

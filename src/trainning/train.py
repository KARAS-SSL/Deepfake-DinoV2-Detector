import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as pth_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from dinov2.models.vision_transformer import vit_small, vit_large
from PIL import Image
import numpy as np  # Add numpy

# Custom imports from the previous code
from model_loader import load_model, get_device, load_and_transform_image

# Create the MLP head that will classify the features from DINO
class DinoMLPClassifier(nn.Module):
    def __init__(self, backbone):
        super(DinoMLPClassifier, self).__init__()
        self.backbone = backbone
        self.mlp_head = nn.Sequential(
            nn.Linear(backbone.embed_dim, 512),  # MLP layer
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.mlp_head(features)
        return output

#-------------------------------------------------------------------------------------------------------------------------------
# Saving 
# Save training details (model, hyperparams, and losses)
def save_training_details(output_folder, model, hyperparams, train_losses, val_losses, best_model_state=None):
    # Save final model weights
    model_path = os.path.join(output_folder, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters
    hyperparams_path = os.path.join(output_folder, 'hyperparams.txt')
    with open(hyperparams_path, 'w') as f:
        for key, value in hyperparams.items():
            f.write(f'{key}: {value}\n')

    # Save losses as .npy files using numpy
    train_loss_path = os.path.join(output_folder, 'train_losses.npy')
    val_loss_path = os.path.join(output_folder, 'val_losses.npy')
    np.save(train_loss_path, np.array(train_losses))
    np.save(val_loss_path, np.array(val_losses))

    # Save best model weights (with lowest validation loss)
    if best_model_state is not None:
        best_model_path = os.path.join(output_folder, 'model_weights_best.pth')
        torch.save(best_model_state, best_model_path)

#-------------------------------------------------------------------------------------------------------------------------------
# Train and Val
        
# Training function
def trainning_loop(model, train_loader, val_loader, device, epochs, lr, output_folder):
    # Create run subfolder for the current training run
    run_folder = create_output_subfolder(output_folder)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Track the best (lowest) validation loss
    best_model_state = None  # To store the model state for the best validation loss

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        accuracy = correct / total * 100
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Validation phase
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Check if current validation loss is the best we've seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state

    # Save final model, hyperparameters, and losses
    hyperparams = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE
    }
    save_training_details(run_folder, model, hyperparams, train_losses, val_losses, best_model_state)

def validating_loop(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return epoch_loss

#-------------------------------------------------------------------------------------------------------------------------------
# TODO: Everything here
def train_model():
    # Load the datasets
    device = get_device()

    # Load the DINOv2 small model as a backbone
    backbone, patch_size = load_model(version="dinov2_small", device=device, inference=True)

    # Create model
    model = DinoMLPClassifier(backbone, NUM_CLASSES).to(device)

    # Load the dataset
    dataset_paths = {"train": train_dataset_path, "val": val_dataset_path} 
    train_loader, val_loader = load_dataset(dataset_paths, IMAGE_SIZE, BATCH_SIZE)

    # Train the model
    train(model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE, output_folder) 

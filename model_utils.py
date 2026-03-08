import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_model(num_classes=2, weights_path=None):
    model = models.resnet18(weights='IMAGENET1K_V1' if weights_path is None else None)

    # Replace the head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Inference: load weights (weights_only=True on PyTorch 2+ avoids pickle-based exploits)
    if weights_path:
        try:
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
    
    return model

def get_dataloaders(data_dir, batch_size=256):
    data_transforms = get_transforms()

    # Multi-process loading (run script with: python deepfake_detection.py)
    num_workers = 4
    use_persistent = num_workers > 0  # avoids respawning workers each epoch
    
    # pin_memory is great for your RTX 4070, but set to False if it keeps hanging
    pin_memory = torch.cuda.is_available()

    phases = ['Train', 'Validation', 'Test']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in phases}

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'Train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
        )
        for x in phases
    }
    return dataloaders
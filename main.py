# Imports
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set up paths of train and test data
train_dir = "data/train"
test_dir = "data/test"

# get the number of CPUs available
NUM_WORKERS = os.cpu_count()


# Create DataLoader Function for loading the training and testing datasets
def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    # Use ImageFolder() to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get Class names
    class_names = train_data.classes

    # turn images into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, test_dataloader, class_names


IMG_SIZE = 300

# Create transforms pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
print(f'Manually created transforms: {manual_transforms}')

BATCH_SIZE = 32

# Create Dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir,
    test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)

print(test_dataloader, test_dataloader, class_names)

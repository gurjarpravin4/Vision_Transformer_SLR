# Imports
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
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


    IMG_SIZE = 288

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

    # Print a random image to verify
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))
    image, label = image_batch[0], label_batch[0]
    print(image.shape, label)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()


    # 1. Create a class which subclasses the nn.Module
    class PatchEmbedding(nn.Module):
        """
        Turns a 2-D input image into a 1-D sequence learnable embedding vector
        Args:
            in_channels (int): No of color channels for input image default=3 (RGB)
            patch_size (int): Size of a single patch default=16
            embedding_dim (int): Size of embedding to turn image into default=768
        """

        # 2. Initialize the class with appropriate variables
        def __init__(
                self,
                in_channels: int = 3,
                patch_size: int = 16,
                embedding_dim: int = 768
        ):
            super.__init__()

            # 3. Create a layer to turn image into patches
            self.patcher = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            )

            # 4. Create a layer to flatten the patch maps into a single dimension
            self.flatten = nn.Flatten(start_dim=2, end_dim=3)
            # Only flatten the feature map dimensions into a single vector

        # 5. Define forward method
        def forward(self, x):
            # Create assertion to check that inputs are of correct shape
            image_resolution = x.shape[-1]
            assert image_resolution % patch_size == 0, (f"input must be divisible by patch size, "
                                                        f"image shape: {image_resolution}, patch size: {patch_size}")
            # Perform forward pass
            x_patched = self.patcher(x)
            x_flattened = self.flatten(x_patched)

            # 6. Make sure output shape has right order
            return x_flattened.permute(0, 2, 1)

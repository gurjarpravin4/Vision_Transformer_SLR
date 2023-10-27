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


    IMG_SIZE = 224

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
    print(f"Image shape: {image.shape}, label: {label}")

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
            super().__init__()

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


    # Set Seeds so that on every machine the same random image is selected
    torch.manual_seed(42)

    # set patch size
    patch_size = 16

    # Print shape of original image tensor and get image dimensions
    print(f"Image tensor shape: {image.shape}")  # This 'image' is from line number 75
    height, width = image.shape[1], image.shape[2]

    # Get image tensor and add batch dimension
    x = image.unsqueeze(0)
    print(f"Input image with batch dimension shape: {x.shape}")

    # Create an instance of patch embedding layer
    patch_embedding_layer = PatchEmbedding(
        in_channels=3,
        patch_size=patch_size,
        embedding_dim=768
    )

    # Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)
    print(f"Patch Embedded Image Shape: {patch_embedding.shape}")

    # Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(
        torch.ones(batch_size, 1, embedding_dimension),
        requires_grad=True
    )  # 'True' to make sure its learnable
    print(f"Class token embedding shape: {class_token.shape}")

    # Prepend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
    print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

    # Create position embedding
    no_of_patches = int((height * width) / patch_size ** 2)
    position_embedding = nn.Parameter(
        torch.ones(1, no_of_patches + 1, embedding_dimension),
        requires_grad=True)

    # Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding
    print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")
    print(patch_embedding_class_token)

    # Create Multi headed attention block
    class MultiHeadedSelfAttentionBlock(nn.Module):
        """
        Creates a Multi headed self attention block (MSA Block)
        """

        # Initialize class with hyperparameters from table 1
        def __int__(
                self,
                embedding_dim: int = 768,
                num_heads: int = 12,
                attn_dropout: float = 0
        ):
            super().__init__()

            # create the normalization layer
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

            # create MSA layer
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )

            # Create a forward method to pass data through the layers
            def forward(self, x):
                x = self.layer_norm(x)
                attn_output, _ = self.multihead_attn(
                    query=x,
                    key=x,
                    value=x,
                    need_weights=False
                )
                return attn_output


    # Create MLP Block
    class MLPBlock(nn.Module):
        """
        Creates a normalization layer with multilayer perceptron block (MLP block)
        """

        # initialize class parameters from table 1 & 3
        def __init__(
                self,
                embedding_dim: int = 768,
                mlp_size: int = 3072,
                dropout: float = 0.1
        ):
            super().__init__()

            # create the normalization layer
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

            # create MLP layers
            self.mlp = nn.Sequential(
                nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=mlp_size, out_features=embedding_dim),
                nn.Dropout(p=dropout)
            )

        # Create a forward method to pass data through layers
        def forward(self, x):
            x = self.layer_norm(x)
            x = self.mlp(x)
            return x


    # Creating a transformer encoder block by combining all the layers
    class TransformerEncoderBlock(nn.Module):
        """
        Creates a transformer encoder block
        """

        # initialize class parameters from table 1 & 3
        def __init__(
                self,
                embedding_dim: int = 768,
                num_heads: int = 12,
                mlp_size: int = 3072,
                mlp_dropout: float = 0.1,
                attn_dropout: float = 0
        ):
            super().__init__()

            # Create MSA Block
            self.msa_block = MultiHeadedSelfAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout
            )

            # Create MLP MLPBlock
            self.mlp_block = MLPBlock(
                embedding_dim=embedding_dim,
                mlp_size=mlp_size,
                dropout=mlp_dropout
            )

        # Create a forward method to pass data through layers
        def forward(self, x):
            # create residual connection for msa block (add input to the output)
            x = self.msa_block(x) + x

            # create residual connection for msa block (add input to the output)
            x = self.mlp_block(x) + x
            return x


    # Building a Vision Transformer
    class ViT(nn.Module):
        """
        Creates a ViT architecture with ViT-Base hyperparamters by default
        """

        # initialize class parameters from table 1 & 3
        def __init__(
                self,
                img_size: int = 224,
                in_channels: int = 3,
                patch_size: int = 16,
                num_transformer_layers: int = 12,
                embedding_dim: int = 768,
                mlp_size: int = 3072,
                num_heads: int = 12,
                attn_dropout: float = 0,
                mlp_dropout: float = 0.1,
                embedding_dropout: float = 0.1,
                num_classes: int = 1000
        ):
            super().__init__()

            # Make sure image size is divisioble by patch size
            assert img_size % patch_size == 0, (f"input must be divisible by patch size, "
                                                f"image shape: {img_size}, patch size: {patch_size}")

            # Calculate the number of patches = h*w/patch_size^2
            self.num_patches = (img_size * img_size) // patch_size ** 2

            # Create learnable class embedding (needs to go at front of sequence patch embeddings)
            self.class_embedding = nn.Parameter(
                data=torch.randn(1, 1, embedding_dim),
                requires_grad=True
            )

            # Create learnable position embedding
            self.position_embedding = nn.Parameter(
                data=torch.randn(1, self.num_patches + 1, embedding_dim),
                requires_grad=True
            )

            # create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)

            # Create patch embedding layer
            self.patch_embedding = PatchEmbedding(
                in_channels=in_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim
            )

            # Create transformer encoder blocks
            # We can stack them using nn.Sequential() NOTE: * means all
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout) for i in range(num_transformer_layers)])

            # Create classifier head
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim, out_features=num_classes)
            )

        # Create a forward method
        def forward(self, x):
            # get batch_size
            batch_size = x.shape[0]

            # Create class token embedding and expand it to match batch size
            class_token = self.class_embedding.expand(batch_size, -1, 1)

            # Create patch embedding
            x = self.patch_embedding(x)

            # Concat class embedding and patch embedding
            x = torch.cat((class_token, x), 1)

            # Add Position embedding to patch embedding
            x = self.position_embedding + x

            # Run embedding dropout
            x = self.embedding_dropout(x)

            # Pass patch, position and class embedding through encoder layers
            x = self.transformer_encoder(x)

            # Put 0 index logit through classifier
            x = self.classifier(x[:, 0])

            return x

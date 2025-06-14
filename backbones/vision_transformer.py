# Implement Vision Transformer based on personal knowledge
# This work is not done on research
# Have been trained on Cifar-10 dataset, accuracy ~ 0.4. Increase the # params to get better performance.
# Implemented by Mikyx-1  - 21/05/2024


import numpy as np
import torch
import torch.nn as nn


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=10):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        """
        Extracts non-overlapping square patches from input images.

        Parameters:
            input_data (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width)
                                       representing a batch of images.

        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, num_patches, patch_dim), where:
                          - num_patches = (height // patch_size) * (width // patch_size)
                          - patch_dim = channels * patch_size * patch_size

        Raises:
            AssertionError: If height or width of the input image is not divisible by the patch size.

        Example:
            >>> input_data = torch.randn(8, 3, 40, 40)
            >>> extractor = PatchExtractor(patch_size=10)
            >>> patches = extractor(input_data)
            >>> patches.shape
            torch.Size([8, 16, 300])
        """
        batch_size, channels, height, width = input_data.size()
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input height ({height}) and width ({width}) are not divisible by patch_size ({self.patch_size})"

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = (
            input_data.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(batch_size, num_patches, -1)
        )

        return patches


class EmbeddingLayer(nn.Module):
    """
    Embeds input image patches and adds learnable positional encodings.

    This layer transforms flattened image patches into a latent vector space
    and incorporates positional information using a simple positional encoding scheme.

    Parameters:
        latent_size (int): The dimensionality of the embedding space (output of the embedding).
        num_patches (int): Total number of image patches per image.
        input_dim (int): the dimensionality of each flattened patch (patch_size * patch_size * channels).

    Forward Input:
        input (torch.Tensor): A tensor of shape (batch_size, num_patches, input_dim) representing flattened image patches.

    Forward Output:
        torch.Tensor: A tensor of shape (batch_size, num_patches, latent_size), representing patch embeddings
                      with positional encoding added.

    Example:
        >>> input = torch.randn(4, 16, 300)
        >>> emb = EmbeddingLayer(latent_size=512, num_patches=16, input_dim=300)
        >>> output = emb(input)
        >>> output.shape
        torch.Size([4, 16, 512])
    """

    def __init__(
        self, latent_size: int = 1024, num_patches: int = 4, input_dim: int = 768
    ):
        super().__init__()

        self.num_patches = num_patches
        self.pos_embedder = nn.Linear(1, latent_size)
        self.input_embedder = nn.Linear(input_dim, latent_size)
        self.positional_information = (
            torch.arange(0, self.num_patches).reshape(1, num_patches, 1).float()
        )

    def forward(self, input):
        N, num_patches, input_dim = input.shape
        input_embedding = self.input_embedder(input)
        positional_embedding = torch.tile(self.positional_information, (N, 1, 1))
        positional_embedding = self.pos_embedder(positional_embedding)
        return positional_embedding + input_embedding


class ViT(nn.Module):
    """
    A simple implementation of the Vision Transformer (ViT) model.

    This model divides the input image into non-overlapping patches,
    embeds them, applies multi-head self-attention and feedforward transformations,
    and performs classification on the resulting representations.

    Parameters:
        patch_size (int): Size of each image patch (patches are square).
        img_dimension (tuple): Tuple of (height, width) representing input image size.
        latent_size (int): Dimensionality of the patch embeddings and transformer layers.
        num_heads (int): Number of heads in the multi-head attention layer.
        num_classes (int): Number of output classes for classification.


    Forward Input:
        x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

    Forward Output:
        torch.Tensor: A tensor of shape (batch_size, num_classes) representing logits.

    Example:
        >>> x = torch.randn(8, 3, 64, 64)
        >>> model = ViT(patch_size=16, img_dimension=(64, 64), latent_size=512, num_heads=4, num_classes=10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([8, 10])
    """

    def __init__(
        self,
        patch_size: int = 16,
        img_dimension: tuple = (32, 32),
        latent_size: int = 1024,
        num_heads: int = 1,
        num_classes: int = 2,
    ):
        super().__init__()

        assert (
            img_dimension[0] % patch_size == 0 and img_dimension[1] % patch_size == 0
        ), "Patch size is not divisible by image dimension !!"

        self.num_patches_h = img_dimension[0] // patch_size
        self.num_patches_w = img_dimension[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patchifier = PatchExtractor(patch_size)
        self.embedding_layer = EmbeddingLayer(
            latent_size=latent_size,
            num_patches=self.num_patches,
            input_dim=patch_size * patch_size * 3,
        )
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=latent_size, num_heads=num_heads
        )
        self.norm_1 = nn.LayerNorm(normalized_shape=latent_size)
        self.norm_2 = nn.LayerNorm(normalized_shape=latent_size)

        self.feed_forward_block = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.GELU(),
            nn.Linear(latent_size * 2, latent_size),
        )

        self.output_layer = nn.Linear(latent_size * self.num_patches, num_classes)

    def forward(self, x):
        x = self.patchifier(x)
        x = self.embedding_layer(x)

        x = self.norm_1(self.multi_head_attn(x, x, x)[0] + x)

        x = self.norm_2(self.feed_forward_block(x) + x)

        x = self.output_layer(x.flatten(start_dim=1))
        return x


if __name__ == "__main__":
    sample = torch.randn((1, 3, 128, 128))

    vit = ViT(
        patch_size=32,
        img_dimension=(128, 128),
        latent_size=1024,
        num_heads=2,
        num_classes=5,
    )
    print(vit(sample).shape)

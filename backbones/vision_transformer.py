# Implement Vision Transformer based on personal knowledge
# This work is not done on research
# Have been trained on Cifar-10 dataset, accuracy ~ 0.4. Increase the # params to get better performance.
# Implemented by Mikyx-1  - 21/05/2024


import torch
import torch.nn as nn
import numpy as np



class PatchExtractor(nn.Module):
    def __init__(self, patch_size = 10):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        batch_size, channels, height, width = input_data.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
        f"Input height ({height}) and width ({width}) are not divisible by patch_size ({self.patch_size})"

        num_patches_h = height//self.patch_size
        num_patches_w = width//self.patch_size
        num_patches = num_patches_h*num_patches_w

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
                  unfold(3, self.patch_size, self.patch_size). \
                  permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, num_patches, -1)
        
        return patches
    
class EmbeddingLayer(nn.Module):
    def __init__(self, latent_size: int = 1024, 
                        num_patches: int = 4, 
                        input_dim: int = 768):
        super().__init__()
        
        self.num_patches = num_patches
        self.pos_embedder = nn.Linear(1, latent_size)
        self.input_embedder = nn.Linear(input_dim, latent_size)
        self.positional_information = torch.arange(0, self.num_patches).\
                                      reshape(1, num_patches, 1).float()

    def forward(self, input):
        N, num_patches, input_dim = input.shape
        input_embedding = self.input_embedder(input)
        positional_embedding = torch.tile(self.positional_information, (N, 1, 1))
        positional_embedding = self.pos_embedder(positional_embedding)
        return positional_embedding + input_embedding
    

class ViT(nn.Module):
    def __init__(self, patch_size: int = 16, 
                       img_dimension: tuple = (32, 32), 
                       latent_size: int = 1024, 
                       num_heads: int = 1, 
                       num_classes: int = 2):
        super().__init__()
        
        assert img_dimension[0]%patch_size == 0 and \
               img_dimension[1]%patch_size == 0, "Patch size is not divisible by image dimension !!"
        

        self.num_patches_h = img_dimension[0]//patch_size
        self.num_patches_w = img_dimension[1]//patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patchifier = PatchExtractor(patch_size)
        self.embedding_layer = EmbeddingLayer(latent_size=latent_size,
                                              num_patches=self.num_patches,
                                              input_dim=patch_size*patch_size*3)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=latent_size, num_heads=num_heads)
        self.norm_1 = nn.LayerNorm(normalized_shape=latent_size)
        self.norm_2 = nn.LayerNorm(normalized_shape=latent_size)

        self.feed_forward_block = nn.Sequential(nn.Linear(latent_size, latent_size*2), 
                                                nn.Linear(latent_size*2, latent_size))
        
        self.output_layer = nn.Linear(latent_size*self.num_patches, num_classes)
    def forward(self, x):
        x = self.patchifier(x)
        x = self.embedding_layer(x)

        x = self.norm_1(self.multi_head_attn(x, x, x)[0] + x)

        x = self.norm_2(self.feed_forward_block(x) + x)

        x = self.output_layer(x.flatten(start_dim=1))
        return x


if __name__ == "__main__":
    sample = torch.randn((1, 3, 128, 128))
    
    vit = ViT(patch_size=32, 
              img_dimension=(128, 128),
              latent_size=1024,
              num_heads=2,
              num_classes=5)
    print(vit(sample).shape)



    

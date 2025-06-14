import torch
from torch import nn
from vision_transformer import EmbeddingLayer, PatchExtractor


def random_masking(x, mask_ratio):
    """
    Applies per-sample random masking to a batch of patch embeddings for masked autoencoder training.

    This function randomly masks a specified ratio of tokens (patch embeddings) from each sample in the batch,
    and returns:
        - The visible (unmasked) tokens,
        - a binary mask indicating which tokens were masked.
        - and an index mapping to restore the original token order after decoding.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D), where:
                          - B is the batch size,
                          - N is the number of patches (tokens) per image,
                          - D is the embedding dimension per patch.

        mask_ratio (float): The fraction of patches to mask (between 0 and 1). For example,
                            0.75 means 75% of the patches will be masked.

    Returns:
        x_masked (torch.Tensor): Tensor of shape (B, N_visible, D), where N_visible = int(N * (1 - mask_ratio)).
                                 Contains only the visible (unmasked) patches.

        mask (torch.Tensor): Tensor of shape (B, N) with binary values:
                             - 0 for visible (unmasked) tokens.
                             - 1 for masked tokens.
                             The mask is ordered to match the original patch positions.

        ids_restore (torch.Tensor): Tensor of shape (B, N) containing indices that can be used to restore the
                                    original token order from the masked + mask tokens in the decoder.

    How it works:
        - Random noise is generated per-sample to shuffle patch indices randomly.
        - The top `N_visible` are selected to keep.
        - The rest are considered masked and replaced with a learnable mask token in the decoder.
        - The `ids_restore` tensor allows restoring the original ordering of all patches
          (both visible and masked) in the decoder.

    Example:
        >>> x = torch.randn(4, 196, 768)  # (batch_size, num_patches, embedding_dim)
        >>> x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)
        >>> x_masked.shape  # Expected: (4, 49, 768)
        torch.Size([4, 49, 768])
        >>> mask.shape      # Expected: (4, 196)
        torch.Size([4, 196])
        >>> ids_restore.shape
        torch.Size([4, 196])
    """

    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)  # Sort out largest probs
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # Return the order

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class MAEEncoder(nn.Module):
    def __init__(self, patch_size, img_size, latent_dim, num_heads):
        super().__init__()
        self.patchify = PatchExtractor(patch_size)
        self.embed = EmbeddingLayer(
            latent_size=latent_dim,
            num_patches=(img_size[0] // patch_size) * (img_size[1] // patch_size),
            input_dim=3 * patch_size * patch_size,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads),
            num_layers=4,
        )

    def forward(self, imgs, mask_ratio=0.75):
        patches = self.patchify(imgs)
        tokens = self.embed(patches)
        x_masked, mask, ids_restore = random_masking(tokens, mask_ratio)
        x_encoded = self.encoder(x_masked)
        return x_encoded, mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, latent_dim, decoder_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.proj = nn.Linear(latent_dim, decoder_dim, bias=False)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8),
            num_layers=4,
        )
        self.output_layer = nn.Linear(decoder_dim, patch_dim)  # patch_dim = C * P * P

    def forward(self, x_encoded, ids_restore):
        B, L, D = x_encoded.shape
        N = ids_restore.shape[1]

        x_proj = self.proj(x_encoded)

        mask_tokens = self.mask_token.expand(B, N - L, -1)
        x_full = torch.cat([x_proj, mask_tokens], dim=1)

        x_unshuffled = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        x_decoded = self.decoder(x_unshuffled)
        return self.output_layer(x_decoded)  # reconstruct patches


class MaskedAutoencoder(nn.Module):
    def __init__(
        self, patch_size=16, img_size=(128, 128), latent_dim=512, decoder_dim=256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_dim = 3 * patch_size * patch_size

        self.encoder = MAEEncoder(patch_size, img_size, latent_dim, num_heads=8)
        self.decoder = MAEDecoder(self.num_patches, latent_dim, decoder_dim)

    def forward(self, imgs, mask_ratio=0.75):
        x_encoded, mask, ids_restore = self.encoder(imgs, mask_ratio)
        x_recon = self.decoder(x_encoded, ids_restore)
        return x_recon, mask


def mae_loss(patches, recon, mask):
    loss = (recon - patches) ** 2
    loss = loss.mean(dim=-1)
    return (loss * mask).sum() / mask.sum()

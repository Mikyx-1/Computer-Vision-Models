import matplotlib.pyplot as plt
import torch
from mae import MaskedAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# === Config ===
BATCH_SIZE = 8
MASK_RATIO = 0.75
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# === Dataset ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Trained MAE ===
model = MaskedAutoencoder(patch_size=16, img_size=(128, 128),
                          latent_dim=512, decoder_dim=256).to(DEVICE)

# Load checkpoint if saved (optional)
model.load_state_dict(torch.load("mae_pretrained.pth"))

model.eval()

# === Run Reconstruction ===
with torch.no_grad():
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(DEVICE)

    # Patchify ground truth
    B, C, H, W = imgs.shape
    patchify = torch.nn.Unfold(kernel_size=16, stride=16).to(DEVICE)
    patches = patchify(imgs).transpose(1, 2)  # (B, N, patch_dim)

    # Model forward
    x_recon, mask = model(imgs, mask_ratio=MASK_RATIO)

    # Reconstructed patches → full image
    x_recon = x_recon.transpose(1, 2)  # (B, patch_dim, N)
    reconstructed = torch.nn.functional.fold(x_recon, output_size=(H, W),
                                             kernel_size=16, stride=16)

    # Masked input image (zero out masked patches)
    visible_mask = 1.0 - mask.unsqueeze(-1)  # (B, N, 1)
    visible_patches = patches * visible_mask
    visible_patches = visible_patches.transpose(1, 2)  # (B, patch_dim, N)
    visible_img = torch.nn.functional.fold(visible_patches, output_size=(H, W),
                                           kernel_size=16, stride=16)

# === Plotting ===
def show_images(img_list, titles):
    img_grid = torch.cat(img_list, dim=0)
    img_grid = utils.make_grid(img_grid, nrow=BATCH_SIZE)
    np_img = img_grid.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 5))
    plt.imshow(np_img)
    plt.title(titles)
    plt.axis("off")
    plt.show()

# Clamp for valid image range [0,1]
reconstructed = reconstructed.clamp(0, 1)
visible_img = visible_img.clamp(0, 1)
imgs = imgs.cpu().clamp(0, 1)

# Show all
show_images([imgs, visible_img.cpu(), reconstructed.cpu()],
            titles="Original | Masked Input | Reconstructed Output")

import torch
from mae import MaskedAutoencoder, mae_loss
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = MaskedAutoencoder(patch_size=16, img_size=(128, 128), 
                          latent_dim=512, decoder_dim=256).to(device)

optimiser = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)


num_epochs = 10
mask_ratio = 0.75

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in loop:
        imgs, _ = batch # ignore labels
        imgs = imgs.to(device)

        # Patchify ground truth
        with torch.no_grad():
            B, C, H, W = imgs.shape
            patches = model.encoder.patchify(imgs)    # (B, N, patch_dim)

        # Forward
        x_recon, mask = model(imgs, mask_ratio)

        # Loss
        loss = mae_loss(patches, x_recon, mask)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "mae_pretrained.pth")
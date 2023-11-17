# Built but not tested

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, iter):
        super().__init__()
        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                 kernel_size=kernel_size, padding=kernel_size-2)]
        for i in range(iter-1):
            self.layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                         kernel_size=kernel_size, padding=kernel_size-2))
            
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.block(x)
    

class VGG19(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.block1 = ConvBlock(3, 64, 3, 2)
        self.block2 = ConvBlock(64, 128, 3, 2)
        self.block3 = ConvBlock(128, 256, 3, 4)
        self.block4 = ConvBlock(256, 512, 3, 4)
        self.block5 = ConvBlock(512, 512, 3, 4)

        self.fc1 = nn.Linear(512*49, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpool(self.relu(self.block1(x)))
        x = self.maxpool(self.relu(self.block2(x)))
        x = self.maxpool(self.relu(self.block3(x)))
        x = self.maxpool(self.relu(self.block4(x)))
        x = self.maxpool(self.relu(self.block5(x)))

        x = x.view(-1, 512*49)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    vgg19 = VGG19(3, 1000)
    a = torch.randn((1, 3, 224, 224))
    print(vgg19(a).shape)
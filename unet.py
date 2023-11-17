# Not done

import torch
from torch import nn


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.down_block1 = nn.Sequential(CBR(in_channels, 64, 3, 1), 
                                         CBR(64, 64, 3, 1))
        
        self.down_block2 = nn.Sequential(CBR(64, 128, 3, 1), 
                                         CBR(128, 128, 3, 1))
        
        self.down_block3 = nn.Sequential(CBR(128, 256, 3, 1), 
                                         CBR(256, 256, 3, 1))
        
        self.down_block4 = nn.Sequential(CBR(256, 512, 3, 1), 
                                         CBR(512, 512, 3, 1))
        
        self.down_block5 = nn.Sequential(CBR(512, 1024, 3, 1), 
                                         CBR(1024, 1024, 3, 1))
        


    def forward(self, input_):
        x1 = self.down_block1(input_)
        x2 = self.down_block2(self.maxpool(x1))
        x3 = self.down_block3(self.maxpool(x2))
        x4 = self.down_block4(self.maxpool(x3))
        x5 = self.down_block5(self.maxpool(x4))

        return x5
    
if __name__ == "__main__":
    unet = Unet(3, 1)
    sample = torch.randn((1, 3, 256, 256))
    print(unet(sample).shape)
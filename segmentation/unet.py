# Tested successfully

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
        

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        
        self.up_block1 = nn.Sequential(CBR(1024, 512, 3, 1), 
                                       CBR(512, 512, 3, 1))
        
        self.up_block2 = nn.Sequential(CBR(512, 256, 3, 1), 
                                       CBR(256, 256, 3, 1))
        
        self.up_block3 = nn.Sequential(CBR(256, 128, 3, 1), 
                                       CBR(128, 128, 3, 1))
        
        self.up_block4 = nn.Sequential(CBR(128, 64, 3, 1),
                                       CBR(64, 64, 3, 1))
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, input_):
        x11 = self.down_block1(input_)
        x21 = self.down_block2(self.maxpool(x11)) # 64 channels
        x31 = self.down_block3(self.maxpool(x21)) # 128 channels
        x41 = self.down_block4(self.maxpool(x31)) # 256 channels
        x51 = self.down_block5(self.maxpool(x41)) # 512 channels

        x52 = self.up_block1(torch.cat([x41, self.up_conv1(x51)], dim=1))
        x42 = self.up_block2(torch.cat([x31, self.up_conv2(x52)], dim =1))
        x32 = self.up_block3(torch.cat([x21, self.up_conv3(x42)], dim=1)) 
        x22 = self.up_block4(torch.cat([x11, self.up_conv4(x32)], dim=1))
        out = self.final_conv(x22)


        return out
    
if __name__ == "__main__":
    unet = Unet(3, 1)
    sample = torch.randn((1, 3, 256, 256))
    print(unet(sample).shape)

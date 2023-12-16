'''
Implementation of SegNet
Inherited from vinceecws  and modified by Mikyx-1
Original Implementation link: https://github.com/vinceecws/SegNet_PyTorch/tree/master
'''

import torch
import torch.nn.functional as F
from torch import nn


class SegNet(nn.Module):
    def __init__(self, in_channel = 3, num_classes = 10):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(2, stride=2)

        self.num_layers = [64, 128, 256, 512, 512]
        encoding_layers = []
        
        current_num_channels = in_channel
        for i in range(len(self.num_layers)):
            encoding_layer = []
            
            encoding_layer += [nn.Conv2d(current_num_channels, self.num_layers[i], kernel_size=3, stride=1, padding=1, bias = False), 
                                  nn.BatchNorm2d(self.num_layers[i]),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.num_layers[i], self.num_layers[i], kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(self.num_layers[i]),
                                  nn.ReLU()]
            if i > 1:
                encoding_layer += [nn.Conv2d(self.num_layers[i], self.num_layers[i], kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(self.num_layers[i]),
                                      nn.ReLU()]
                
            encoding_layer = nn.Sequential(*encoding_layer)
            encoding_layers.append(encoding_layer)
            current_num_channels = self.num_layers[i]

        self.encoding_block = nn.Sequential(*encoding_layers)
        del encoding_layers
        del encoding_layer

        decoding_layers = []

        self.num_layers.insert(0, num_classes)
        for i in range(len(self.num_layers)-1):
            decoding_layer = []
            decoding_layer += [nn.Conv2d(current_num_channels, self.num_layers[-i-1], kernel_size=3, stride=1, padding=1, bias=False),
                               nn.BatchNorm2d(self.num_layers[-i-1]),
                               nn.ReLU()]
            
            if i < 3:
                decoding_layer += [nn.Conv2d(self.num_layers[-i-1], self.num_layers[-i-1], kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.num_layers[-i-1]),
                                   nn.ReLU(),
                                   nn.Conv2d(self.num_layers[-i-1], self.num_layers[-i-2], kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.num_layers[-i-2]),
                                   nn.ReLU()]
                
            else:
                decoding_layer += [nn.Conv2d(self.num_layers[-i-1], self.num_layers[-i-2], kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.num_layers[-i-2]),
                                   nn.ReLU()]
                
            current_num_channels = self.num_layers[-i-2]

            decoding_layer = nn.Sequential(*decoding_layer)
            decoding_layers.append(decoding_layer)
        self.decoding_block = nn.Sequential(*decoding_layers)


    def forward(self, input_):
        # Encoding Stage 1
        x1 = self.encoding_block[0](input_)
        x1, ind1 = self.maxpool(x1)

        # Encoding Stage 2
        x2, ind2 = self.maxpool(self.encoding_block[1](x1))

        # Encoding Stage 3
        x3, ind3 = self.maxpool(self.encoding_block[2](x2))

        # Encoding Stage 4
        x4, ind4 = self.maxpool(self.encoding_block[3](x3))

        # Encoding Stage 5
        x5, ind5 = self.maxpool(self.encoding_block[4](x4))

        # Decoding Stage 1
        y1 = self.decoding_block[0](self.max_unpool(x5, ind5))

        # Decoding Stage 2
        y2 = self.decoding_block[1](self.max_unpool(y1, ind4))

        # Decoding Stage 3
        y3 = self.decoding_block[2](self.max_unpool(y2, ind3))

        # Decoding Stage 4
        y4 = self.decoding_block[3](self.max_unpool(y3, ind2))

        # Decoding Stage 3
        y5 = self.decoding_block[4](self.max_unpool(y4, ind1))

        return y5
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegNet().to(device)

    print(model(torch.randn((1, 3, 224, 224))).to(device).shape)

                

                
            


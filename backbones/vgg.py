'''
Implementation of original VGG network by Mikyx-1 
From paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
Link paper: https://arxiv.org/pdf/1409.1556.pdf

Configurations, settings, and inference checked
'''

import torch
from torch import nn

torch.set_grad_enabled(False)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A-LRN': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super().__init__()
        if vgg_name not in cfg:
            raise "Architecture not included, please try again"
        
        self.layer_names = cfg[vgg_name]
        layers = [nn.Conv2d(3, self.layer_names[0], kernel_size=3, padding = 1)]
        if "LRN" in vgg_name:
            layers += [nn.LocalResponseNorm(5)]

        layers += [nn.ReLU()]

        last_num_layer = self.layer_names[0]
        for ith, layer_name in enumerate(self.layer_names[1:]):
            if isinstance(layer_name, int):
                if vgg_name == 'C' and ith >= 6 and self.layer_names[ith+2] == 'M':
                    layers += [nn.Conv2d(last_num_layer, layer_name, kernel_size=1, stride=1, padding = 0)]
                else:
                    layers += [nn.Conv2d(last_num_layer, layer_name, kernel_size=3, stride=1, padding = 1)]
    
                layers += [nn.ReLU()]
                last_num_layer = layer_name
            else:
                layers.append(nn.MaxPool2d(2, 2))

        self.layers = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_num_layer, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
        


if __name__ == "__main__":
    modelA = VGG("A", 10)
    modelALRN = VGG("A-LRN", 10)
    modelB = VGG('B', 10)
    modelC = VGG("C", 10)
    modelD = VGG("D", 10)
    modelE = VGG('E', 10)

    dummy = torch.randn((1, 3, 224, 224))
    print(f"Model A output shape: {modelA(dummy).shape}")
    print(f"Model A-LRN output shape: {modelALRN(dummy).shape}")
    print(f"Model B output shape: {modelB(dummy).shape}")
    print(f"Model C output shape: {modelC(dummy).shape}")
    print(f"Model D output shape: {modelD(dummy).shape}")
    print(f"Model E output shape: {modelE(dummy).shape}")




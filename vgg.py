"""
VGG Implementation
Not tested
"""

import torch
from torch import nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super().__init__()
        if vgg_name not in cfg:
            raise "Architecture not included"
        
        self.layer_names = cfg[vgg_name]
        layers = [nn.Conv2d(3, self.layer_names[0], kernel_size=3, padding=1)]
        last_num_layer = self.layer_names[0]
        for layer_name in self.layer_names:
            if isinstance(layer_name, int):
                layers += [nn.Conv2d(last_num_layer, layer_name, kernel_size=3, stride=1, padding=1), 
                           nn.BatchNorm2d(layer_name), nn.ReLU()]
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
        x = self.fc(x)
        return x
    

def VGG11():
    return VGG("VGG11", 10)

def VGG13():
    return VGG("VGG13", 10)

def VGG16():
    return VGG("VGG16", 10)

def VGG19():
    return VGG("VGG19", 10)

def test():
    a = torch.randn((1, 3, 224, 224))
    model = VGG19()
    print(model(a).shape)

if __name__ == "__main__":
    test()
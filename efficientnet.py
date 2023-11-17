import torch
import torch.nn as nn
from math import ceil
from torchvision.ops import stochastic_depth

base_model = [
    [1, 16, 1, 1, 3],              # expand ratio, channels, repeats, stride, kernel_size
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    "b0":(0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                groups=1):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        self.block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                       self.stride, self.padding, bias=False),
                             nn.BatchNorm2d(self.out_channels),
                             nn.SiLU())
    def forward(self, x):
        return self.block(x)


class SE(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SE, self).__init__()
        self.in_channels = in_channels
        self.reduced_dim = reduced_dim
        
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               nn.Conv2d(self.in_channels, self.reduced_dim, kernel_size=1,
                                        stride=1),
                               nn.SiLU(),
                               nn.Conv2d(self.reduced_dim, self.in_channels, kernel_size=1,
                                        stride=1),
                               nn.Sigmoid())
        
    def forward(self, x):
        return x*self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, 
                stride, kernel_size, padding, reduction=4):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.reduction = reduction
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.hidden_dim = self.in_channels*self.expand_ratio
        self.reduced_dim = self.hidden_dim//self.reduction
        self.expand = self.hidden_dim != self.in_channels
        self.use_residual = self.in_channels == self.out_channels and stride==1
        
        if self.expand:
            self.expand_conv = CNNBlock(in_channels=self.in_channels, out_channels = self.hidden_dim,
                                       kernel_size=3, stride=1, padding=1)
        self.conv = nn.Sequential(CNNBlock(in_channels = self.hidden_dim, out_channels=self.hidden_dim,
                                          stride=self.stride, kernel_size=self.kernel_size, padding=self.padding,
                                          groups=self.hidden_dim),
                                 SE(self.hidden_dim, self.reduced_dim),
                                 nn.Conv2d(self.hidden_dim, self.out_channels, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(self.out_channels))
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return stochastic_depth(self.conv(x), p=0.6, mode="row", training=True) + inputs
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        self.width_factor, self.depth_factor, self.dropout_rate = self.calculate_factors(version)
        self.last_channels = ceil(1280*self.width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(self.width_factor, self.depth_factor, self.last_channels)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout_rate),
                                       nn.Linear(self.last_channels, num_classes))
        
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, version, alpha=1.2, beta=1.1):
        channels = int(32*self.width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*self.width_factor)/4)
            layers_repeats = ceil(repeats*self.depth_factor)
            for layer in range(layers_repeats):
                features.append(MBConv(in_channels=in_channels, out_channels=out_channels,
                                                     expand_ratio = expand_ratio, stride=stride if layer==0 else 1,
                                                     kernel_size=kernel_size, padding=kernel_size//2))
                in_channels = out_channels
        features.append(CNNBlock(in_channels, self.last_channels, kernel_size=1,
                                stride=1, padding=0))
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        x = self.classifier(x.view(x.shape[0], -1))
        return x


if __name__ == "__main__":
    version = "b0"    # default
    num_categories = 1000      # according to imagenet1K dataset
    model = EfficientNet(version=version, num_classes = num_categories)
    a = torch.randn((1, 3, 256, 256))
    print(model(a).shape)
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models


class Atrous_Convolution(nn.Module):
    def __init__(
            self, input_channels, kernel_size, pad, dilation_rate,
            output_channels=256):
        super(Atrous_Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1x1 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, 
                                           kernel_size=1, pad = 0, dilation_rate=1)
        self.conv_6x6 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels,
                                           kernel_size = 3, pad=6, dilation_rate=6)
        self.conv_12x12 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, 
                                             kernel_size=3, pad=12, dilation_rate=12)
        self.conv_18x18 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels,
                                             kernel_size=3, pad=18, dilation_rate=18)
        
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                  kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        
        self.final_conv = Atrous_Convolution(input_channels=out_channels*5, output_channels=out_channels, 
                                             kernel_size=1, pad = 0, dilation_rate=1)
        
    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(img_pool_opt, size = x_18x18.size()[2:], mode="bilinear", align_corners=True)
        concat = torch.cat([x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt], dim=1)
        return self.final_conv(concat)



class ResNet_50(nn.Module):
    def __init__(self, output_layer=None):
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x


class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ResNet_50("layer3")  # 
        self.low_level_features = ResNet_50("layer1")   # 
        self.assp = ASSP(in_channels=1024, out_channels=256)
        self.conv_1x1 = Atrous_Convolution(input_channels=256, output_channels=48, kernel_size=1, 
                                           dilation_rate=1, pad=0)
        self.conv_3x3 = nn.Sequential(nn.Conv2d(304, 256, 3, padding = 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):

        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_conv1x1 = self.conv_1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)
        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_out = self.classifier(x_3x3_upscaled)
        return x_out

if __name__ == "__main__":
    model = Deeplabv3Plus(2)
    model.eval()

    print(model(torch.randn((1, 3, 256, 256))).shape)
'''
Implementation of FastFCN
Inherited from HuiKai Wu and modified by Mikyx-1

Comment: This code use RAW IMPLEMENTATION of ResNet from pytorch instead of re-writing resnet with different dilation rate
Reference: https://arxiv.org/abs/1903.11816
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
    

class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None):
        super().__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        
    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode="bilinear", align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode="bilinear", align_corners=True)

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], 1)
        return inputs[0], inputs[1], inputs[2], feat
    

class BaseNet(nn.Module):
    def __init__(self, nclass, aux, se_loss, backbone="resnet50", norm_layer=None):
        super().__init__()
        self.n_class = nclass
        self.aux = aux
        self.se_loss = se_loss
        
        if backbone == "resnet50": 
            self.pretrained = models.resnet50(pretrained=False)
        elif backbone == "resnet101":
            self.pretrained = models.resnet101(pretrained=False)
        elif backbone == "resnet152":
            self.pretrained = models.resnet152(pretrained=False)

        else:
            raise RuntimeError("Not supported: {}".format(backbone))
        
        self.backbone = backbone
        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer)

    
    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return self.jpu(c1, c2, c3, c4)
    
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
    

class FCN(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, training=True):
        super(FCN, self).__init__(nclass, aux=aux, se_loss=se_loss, backbone=backbone, norm_layer=norm_layer)
        self.head = FCNHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

        self.training = training

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = F.interpolate(x, imsize, mode="bilinear", align_corners=True)
        outputs = [x]
        if self.aux and self.training:  # For calculating auxiliary loss
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, imsize, mode="bilinear", align_corners=True)
            outputs.append(auxout)
            return outputs
        return outputs[0]
    

if __name__ == "__main__":
    device = "cpu"
    model = FCN(nclass=2, backbone="resnet50", aux=True, se_loss=False, training=True)
    print(len(model(torch.randn((1, 3, 448, 448)))))

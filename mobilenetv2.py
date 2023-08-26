# MobileNet V2 
# Tested


import torch
from torch import nn, optim


class InvRes(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, device):
        super().__init__()
        ## Stage 1
        self.conv1 = nn.Conv2d(in_channels, expand_ratio*in_channels, kernel_size= 1, 
                               stride=1, bias = False, device=device)
        self.bn1 = nn.BatchNorm2d(expand_ratio*in_channels, device=device)
        
        ## Stage 2
        self.conv2 = nn.Conv2d(expand_ratio*in_channels, expand_ratio*in_channels, 
                               kernel_size = 3, stride=stride, padding=0 if (stride==0) else 1, 
                               groups=expand_ratio*in_channels, bias = False, device=device)
        self.bn2 = nn.BatchNorm2d(expand_ratio*in_channels, device=device)

        ## Stage 3

        self.conv3 = nn.Conv2d(expand_ratio*in_channels, out_channels, kernel_size=1, 
                               stride=1, bias = False, device=device)
        self.bn3 = nn.BatchNorm2d(out_channels, device=device)

        self.use_add_operation = True if (in_channels==out_channels and stride ==1) else False
    def forward(self, input_):
        # Stage 1
        stage1 = nn.ReLU6()(self.bn1(self.conv1(input_)))

        # Stage 2
        stage2 = nn.ReLU6()(self.bn2(self.conv2(stage1)))

        # Stage 3
        stage3 = self.bn3(self.conv3(stage2))

        if self.use_add_operation:
            return stage3 + input_
        else:
            return stage3
        


class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, device):
        super().__init__()
        self.hyper_params = [[1, 16, 1, 1], # expand_ratio, num_channels, iters, stride
                             [6, 24, 2, 2],
                             [6, 32, 3, 2],
                             [6, 64, 4, 2],
                             [6, 96, 3, 1],
                             [6, 160, 3, 2],
                             [6, 320, 1, 1]]
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, 
                               kernel_size=3, stride=2, padding=0, device=device)
        
        layers = []
        in_channels = 32
        for i in range(len(self.hyper_params)):
            for j in range(self.hyper_params[i][2]):
                out_channels = self.hyper_params[i][1]
                expand_ratio = self.hyper_params[i][0]
                stride = self.hyper_params[i][3]

                if (j != self.hyper_params[i][2]-1):
                    layers += [InvRes(in_channels=in_channels, out_channels=out_channels, 
                                        expand_ratio=expand_ratio, stride=1, device=device)]
                else:
                    layers += [InvRes(in_channels=in_channels, out_channels=out_channels, 
                                        expand_ratio=expand_ratio, stride=stride, device=device)]


                in_channels = out_channels
        self.backbone = nn.Sequential(*layers)

        self.conv2 = nn.Conv2d(self.hyper_params[-1][1], 1280, kernel_size=1, stride=1,
                               bias=True, device = device)
        
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, num_classes, device=device)

    def forward(self, input_):
        head = self.conv1(input_)
        backbone = self.backbone(head)

        tail = self.avgPool(self.conv2(backbone)).squeeze(-1).squeeze(-1)
        output = self.fc1(tail)
        return output



        

if __name__ == "__main__":
    model = MobileNetV2(in_channels=3, num_classes=10, device="cpu")
    a = torch.randn((1, 3, 224, 224))
    res = model(a)
    print(res.shape)
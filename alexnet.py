# Built but not tested

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, 
                               stride=4)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=256*9, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.classifier = nn.Linear(in_features=4096, out_features= num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))

        x = x.view(-1, 256*9)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model = AlexNet(3, 1000)
    a = torch.randn((1, 3, 224, 224))
    print(model(a).shape)

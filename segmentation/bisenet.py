'''
Implementation of BiSeNetV1
'''


from torch import nn, optim
import torch
import torch.nn.functional as F
from torchvision import models

torch.cuda.empty_cache()

# Define the base Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # Head of block is a convulution layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # After conv layer is the batch noarmalization layer 
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Tail of this block is the ReLU function 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Main forward of this block 
        x = self.conv1(x)
        x = self.batch_norm(x)
        return self.relu(x)
    

# Define the Spatial Path with 3 layers of ConvBlock 
class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
    

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        
    def forward(self, x_input):
        # Apply Global Average Pooling
        x = self.avg_pool(x_input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        
        # Channel of x_input and x must be same 
        return torch.mul(x_input, x)



# Define Feature Fusion Module 
class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            
    def forward(self, x_input_1, x_input_2):
        x = torch.cat((x_input_1, x_input_2), dim=1)   # Stack the layers depthwise
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.conv_block(x)      # N x num_classes x H x W
        
        # Apply above branch in feature 
        x = self.avg_pool(feature)        # N x num_classes x 1 x 1
        x = self.relu(self.conv1(x))      # N x num_classes x 1 x 1
        x = self.sigmoid(self.conv2(x))   # N x num_classes x 1 x 1
        
        # Multipy feature and x 
        x = torch.mul(feature, x)         # N x num_classes x H x W
        
        # Combine feature and x
        return torch.add(feature, x)      # N x num_classes x H x W
    
# Build context path 
class ContextPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.max_pool = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    def forward(self, x_input):
        # Get feature from lightweight backbone network
        x = self.conv1(x_input)       # N x H//2 x W//2 x C
        x = self.relu(self.bn1(x))
        x = self.max_pool(x)          # N x H//4 x W//4 x C
        
        # Downsample 1/4
        feature1 = self.layer1(x)     # N x H//4 x W//4 x C
        
        # Downsample 1/8
        feature2 = self.layer2(feature1)  # N x H//8 x W//8 x C
        
        # Downsample 1/16
        feature3 = self.layer3(feature2)  # N x H//16 x W//16 x C
        
        # Downsample 1/32
        feature4 = self.layer4(feature3)  # N x H//32 x W//32 x C
        
        # Build tail with global averange pooling 
        tail = self.avg_pool(feature4)  # N x H//64 x W//64 x C
        return feature3, feature4, tail
    
class BiSeNet(nn.Module):
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.spatial_path = SpatialPath()      
        self.context_path = ContextPath()          #
        self.arm1 = AttentionRefinementModule(in_channels=256, out_channels=256)
        self.arm2 = AttentionRefinementModule(in_channels=512, out_channels=512)
        
        # Supervision for calculate loss 
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        
        # Feature fusion module 
        self.ffm = FeatureFusionModule(num_classes=num_classes, in_channels=1024)
        
        # Final convolution 
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x_input):
        # Spatial path output
        sp_out = self.spatial_path(x_input)  # N x H//8 x W//8 x 256
        
        # Context path output
        feature1, feature2, tail = self.context_path(x_input) # N x H//16 x W//16 x C, N x H//32 x W//32 x C, N x H//64 x W//64 x C
        
#         # apply attention refinement module 
        
        feature1, feature2 = self.arm1(feature1), self.arm2(feature2)
        
#         # Combine output of lightweight model with tail 
        feature2.mul_(tail)
        
#         # Up sampling 
        size2d_out = sp_out.size()[-2:]       # Takes the height and width
        feature1 = F.interpolate(feature1, size=size2d_out, mode='bilinear')   # Resize to the spatial_output
        feature2 = F.interpolate(feature2, size=size2d_out, mode='bilinear')
        context_out = torch.cat((feature1, feature2), dim=1)     # Concatenate the layers depthwise
        
#         # Apply Feature Fusion Module 
        combine_feature = self.ffm(sp_out, context_out)      # What does ffm do ?
        
#         # Up sampling 
        bisenet_out = F.interpolate(combine_feature, scale_factor=8, mode='bilinear')
        bisenet_out = self.conv(bisenet_out)
        
#         # When training model 
        if self.training is True:             
            feature1_sup = self.supervision1(feature1)
            feature2_sup = self.supervision2(feature2)
            feature1_sup = F.interpolate(feature1_sup, size=x_input.size()[-2:], mode='bilinear')
            feature2_sup = F.interpolate(feature2_sup, size=x_input.size()[-2:], mode='bilinear')        
            return bisenet_out, feature1_sup, feature2_sup
        return bisenet_out
    

if __name__ == "__main__":
    model = BiSeNet(num_classes=2, training=False)
    model.eval()
    print(model(torch.randn((1, 3, 288, 288))).shape)
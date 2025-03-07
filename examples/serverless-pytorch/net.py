"""
A simple convnet that works for small resolution image datasets like CIFAR-10.
"""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Resnet style residual block."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + self.residual(x)
        return x


class ResNet18(nn.Module):
    """ResNet-18 model."""
    def __init__(self, num_classes: int = 10, small_resolution: bool = False) -> None:
        import torchvision.models as models

        super(ResNet18, self).__init__()
        # weights = models.ResNet18_Weights.DEFAULT
        weights = None
        self.resnet18 = models.resnet18(weights=weights, progress=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        # Treatment for small resolution images
        if small_resolution:
            self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()

    def forward(self, x):
        return self.resnet18(x)


class SimpleConvNet(nn.Module):
    """
    Similar to ResNet-18, though the stride is set to 1. No pooling layers are used.
    A simple residual block is used.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.residual = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def _residual_block(self, x):
        """Resnet style residual block.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x + self.residual(x)
        return x
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x + self.residual(x)
        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
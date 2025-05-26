"""
A simple convnet that works for small resolution image datasets like CIFAR-10.
"""

import torch.nn as nn


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

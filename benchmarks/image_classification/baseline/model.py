import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetResnet18(nn.Module):
    """A ResNet18 adapted to CIFAR10."""

    def __init__(
        self, num_classes: int = 10, device: str = "cuda", groupnorm: bool = False
    ) -> None:
        super(NetResnet18, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # As the LEAF people do
        # self.net = resnet18(num_classes=10, norm_layer=lambda x: nn.GroupNorm(2, x))
        self.net = resnet18(num_classes=self.num_classes)
        # replace w/ smaller input layer
        self.net.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.net.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = torch.nn.Identity()

    def forward(self, x):
        """Implement forward method."""
        return self.net(x)

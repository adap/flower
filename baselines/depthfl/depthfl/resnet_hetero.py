"""ResNet18 for HeteroFL."""

import numpy as np
import torch.nn as nn


class Scaler(nn.Module):
    """Scaler module for HeteroFL."""

    def __init__(self, rate, scale):
        super().__init__()
        if scale:
            self.rate = rate
        else:
            self.rate = 1

    def forward(self, x):
        """Scaler forward."""
        output = x / self.rate if self.training else x
        return output


class MyBatchNorm(nn.Module):
    """Static Batch Normalization for HeteroFL."""

    def __init__(self, num_channels, track=True):
        super().__init__()
        # change num_groups to 32
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=track)

    def forward(self, x):
        """BatchNorm forward."""
        x = self.norm(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """Convolution layer 3x3."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, planes, stride=1):
    """Convolution layer 1x1."""
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Basic Block for ResNet18."""

    expansion = 1

    def __init__(  # pylint: disable=too-many-arguments
        self,
        inplanes,
        planes,
        stride=1,
        scaler_rate=1,
        downsample=None,
        track=True,
        scale=True,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.scaler = Scaler(scaler_rate, scale)
        self.bn1 = MyBatchNorm(planes, track)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MyBatchNorm(planes, track)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """BasicBlock forward."""
        residual = x

        output = self.conv1(x)
        output = self.scaler(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.scaler(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)
        return output


class Resnet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Resnet model."""

    def __init__(  # pylint: disable=too-many-arguments
        self, hidden_size, block, layers, num_classes, scaler_rate, track, scale
    ):
        super().__init__()

        self.inplanes = hidden_size[0]
        self.norm_layer = MyBatchNorm
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler = Scaler(scaler_rate, scale)
        self.bn1 = self.norm_layer(self.inplanes, track)

        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            hidden_size[0],
            layers[0],
            scaler_rate=scaler_rate,
            track=track,
            scale=scale,
        )
        self.layer2 = self._make_layer(
            block,
            hidden_size[1],
            layers[1],
            stride=2,
            scaler_rate=scaler_rate,
            track=track,
            scale=scale,
        )
        self.layer3 = self._make_layer(
            block,
            hidden_size[2],
            layers[2],
            stride=2,
            scaler_rate=scaler_rate,
            track=track,
            scale=scale,
        )
        self.layer4 = self._make_layer(
            block,
            hidden_size[3],
            layers[3],
            stride=2,
            scaler_rate=scaler_rate,
            track=track,
            scale=scale,
        )
        self.fc_layer = nn.Linear(hidden_size[3] * block.expansion, num_classes)
        self.scala = nn.AdaptiveAvgPool2d(1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(  # pylint: disable=too-many-arguments
        self, block, planes, layers, stride=1, scaler_rate=1, track=True, scale=True
    ):
        """Create a block with layers.

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block.
            scaler_rate (float): for scaler module
            track (bool): static batch normalization
            scale (bool): for scaler module.
        """
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track),
            )
        layer = []
        layer.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                scaler_rate=scaler_rate,
                downsample=downsample,
                track=track,
                scale=scale,
            )
        )
        self.inplanes = planes * block.expansion
        for _i in range(1, layers):
            layer.append(
                block(
                    self.inplanes,
                    planes,
                    scaler_rate=scaler_rate,
                    track=track,
                    scale=scale,
                )
            )

        return nn.Sequential(*layer)

    def forward(self, x):
        """Resnet forward."""
        x = self.conv1(x)
        x = self.scaler(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.scala(x).view(x.size(0), -1)
        out = self.fc_layer(out)

        return [out]


def resnet18(n_blocks=4, track=False, scale=True, num_classes=100):
    """Create resnet18 for HeteroFL.

    Parameters
    ----------
    n_blocks: int
        corresponds to width (divided by 4)
    track: bool
        static batch normalization
    scale: bool
        scaler module
    num_classes: int
        # of labels

    Returns
    -------
    Callable [ [List[int],nn.Module,List[int],int,float,bool,bool], nn.Module]
    """
    # width pruning ratio : (0.25, 0.50, 0.75, 0.10)
    model_rate = n_blocks / 4
    classes_size = num_classes

    hidden_size = [64, 128, 256, 512]
    hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_size]

    scaler_rate = model_rate

    return Resnet(
        hidden_size,
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=classes_size,
        scaler_rate=scaler_rate,
        track=track,
        scale=scale,
    )


# if __name__ == "__main__":
#     from ptflops import get_model_complexity_info

#     model = resnet18(100, 1.0)

#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(
#             model,
#             (3, 32, 32),
#             as_strings=True,
#             print_per_layer_stat=False,
#             verbose=True,
#             units="MMac",
#         )

#         print("{:<30}  {:<8}".format("Computational complexity: ", macs))
#         print("{:<30}  {:<8}".format("Number of parameters: ", params))

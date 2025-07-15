"""ResNet18 for DepthFL."""

import torch.nn as nn


class MyGroupNorm(nn.Module):
    """Group Normalization layer."""

    def __init__(self, num_channels):
        super().__init__()
        # change num_groups to 32
        self.norm = nn.GroupNorm(
            num_groups=16, num_channels=num_channels, eps=1e-5, affine=True
        )

    def forward(self, x):
        """GN forward."""
        x = self.norm(x)
        return x


class MyBatchNorm(nn.Module):
    """Batch Normalization layer."""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=True)

    def forward(self, x):
        """BN forward."""
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


class SepConv(nn.Module):
    """Bottleneck layer module."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        channel_in,
        channel_out,
        kernel_size=3,
        stride=2,
        padding=1,
        norm_layer=MyGroupNorm,
    ):
        super().__init__()
        self.operations = nn.Sequential(
            nn.Conv2d(
                channel_in,
                channel_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=channel_in,
                bias=False,
            ),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channel_in,
                channel_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=channel_in,
                bias=False,
            ),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        """SepConv forward."""
        return self.operations(x)


class BasicBlock(nn.Module):
    """Basic Block for ResNet18."""

    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, norm_layer=None
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """BasicBlock forward."""
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)
        return output


class MultiResnet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Resnet model.

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        n_blocks (int) : Depth of network
        num_classes (int): class num.
        norm_layer (class): type of normalization layer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        block,
        layers,
        n_blocks,
        num_classes=1000,
        norm_layer=MyBatchNorm,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.inplanes = 64
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.feature_fc1 = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion,
                norm_layer=norm_layer,
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
                norm_layer=norm_layer,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
                norm_layer=norm_layer,
            ),
            nn.AdaptiveAvgPool2d(1),
        )

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=64 * block.expansion,
                norm_layer=norm_layer,
            ),
            norm_layer(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

        if n_blocks > 1:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)
            # self.feature_fc2 = nn.Linear(512 * block.expansion, 512 * block.expansion)
            self.scala2 = nn.Sequential(
                SepConv(
                    channel_in=128 * block.expansion,
                    channel_out=256 * block.expansion,
                    norm_layer=norm_layer,
                ),
                SepConv(
                    channel_in=256 * block.expansion,
                    channel_out=512 * block.expansion,
                    norm_layer=norm_layer,
                ),
                nn.AdaptiveAvgPool2d(1),
            )
            self.attention2 = nn.Sequential(
                SepConv(
                    channel_in=128 * block.expansion,
                    channel_out=128 * block.expansion,
                    norm_layer=norm_layer,
                ),
                norm_layer(128 * block.expansion),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Sigmoid(),
            )

        if n_blocks > 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)
            # self.feature_fc3 = nn.Linear(512 * block.expansion, 512 * block.expansion)
            self.scala3 = nn.Sequential(
                SepConv(
                    channel_in=256 * block.expansion,
                    channel_out=512 * block.expansion,
                    norm_layer=norm_layer,
                ),
                nn.AdaptiveAvgPool2d(1),
            )
            self.attention3 = nn.Sequential(
                SepConv(
                    channel_in=256 * block.expansion,
                    channel_out=256 * block.expansion,
                    norm_layer=norm_layer,
                ),
                norm_layer(256 * block.expansion),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Sigmoid(),
            )

        if n_blocks > 3:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.fc_layer = nn.Linear(512 * block.expansion, num_classes)
            self.scala4 = nn.AdaptiveAvgPool2d(1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self, block, planes, layers, stride=1, norm_layer=None
    ):  # pylint: disable=too-many-arguments
        """Create a block with layers.

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block.
            norm_layer (class): type of normalization layer.
        """
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layer = []
        layer.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _i in range(1, layers):
            layer.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layer)

    def forward(self, x):
        """Resnet forward."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        fea1 = self.attention1(x)
        fea1 = fea1 * x
        out1_feature = self.scala1(fea1).view(x.size(0), -1)
        middle_output1 = self.middle_fc1(out1_feature)
        # out1_feature = self.feature_fc1(out1_feature)

        if self.n_blocks == 1:
            return [middle_output1]

        x = self.layer2(x)
        fea2 = self.attention2(x)
        fea2 = fea2 * x
        out2_feature = self.scala2(fea2).view(x.size(0), -1)
        middle_output2 = self.middle_fc2(out2_feature)
        # out2_feature = self.feature_fc2(out2_feature)
        if self.n_blocks == 2:
            return [middle_output1, middle_output2]

        x = self.layer3(x)
        fea3 = self.attention3(x)
        fea3 = fea3 * x
        out3_feature = self.scala3(fea3).view(x.size(0), -1)
        middle_output3 = self.middle_fc3(out3_feature)
        # out3_feature = self.feature_fc3(out3_feature)

        if self.n_blocks == 3:
            return [middle_output1, middle_output2, middle_output3]

        x = self.layer4(x)
        out4_feature = self.scala4(x).view(x.size(0), -1)
        output4 = self.fc_layer(out4_feature)

        return [middle_output1, middle_output2, middle_output3, output4]


def multi_resnet18(n_blocks=1, norm="bn", num_classes=100):
    """Create resnet18 for HeteroFL.

    Parameters
    ----------
    n_blocks: int
        depth of network
    norm: str
        normalization layer type
    num_classes: int
        # of labels

    Returns
    -------
    Callable [ [nn.Module,List[int],int,int,nn.Module], nn.Module]
    """
    if norm == "gn":
        norm_layer = MyGroupNorm

    elif norm == "bn":
        norm_layer = MyBatchNorm

    return MultiResnet(
        BasicBlock,
        [2, 2, 2, 2],
        n_blocks,
        num_classes=num_classes,
        norm_layer=norm_layer,
    )


# if __name__ == "__main__":
#     from ptflops import get_model_complexity_info

#     model = MultiResnet18(n_blocks=4, num_classes=100)

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

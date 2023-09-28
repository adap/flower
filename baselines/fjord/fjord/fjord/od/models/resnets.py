# Adapted from:
#   https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch.nn as nn
import torch.nn.functional as F

from .utils import create_bn_layer, create_conv_layer, \
    create_linear_layer, SequentialWithSampler
from fjord.od.samplers import BaseSampler


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, od, p_s, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.od = od
        self.conv1 = create_conv_layer(
            od, True, in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = create_bn_layer(od, p_s, planes)
        self.conv2 = create_conv_layer(
            od, True, planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = create_bn_layer(od, p_s, planes)

        self.shortcut = SequentialWithSampler()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = SequentialWithSampler(
                create_conv_layer(
                    od, True, in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride, bias=False),
                create_bn_layer(od, p_s, self.expansion*planes)
            )

    def forward(self, x, sampler):
        if sampler is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.bn2(self.conv2(out, p=sampler()))
            shortcut = self.shortcut(x, sampler=sampler)
            assert shortcut.shape == out.shape, \
                f"Shortcut shape: {shortcut.shape} out.shape: {out.shape}"
            out += shortcut
            # out += self.shortcut(x, sampler=sampler)
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, od, p_s, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.od = od
        self.conv1 = create_conv_layer(
            od, True, in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = create_bn_layer(od, p_s, planes)
        self.conv2 = create_conv_layer(
            od, True, planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = create_bn_layer(
            od, p_s, planes)
        self.conv3 = create_conv_layer(
            od, True, planes, self.expansion *
            planes, kernel_size=1, bias=False)
        self.bn3 = create_bn_layer(od, p_s, self.expansion*planes)

        self.shortcut = SequentialWithSampler()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = SequentialWithSampler(
                create_conv_layer(
                    od, True, in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride, bias=False),
                create_bn_layer(od, p_s, self.expansion*planes)
            )

    def forward(self, x, sampler=None):
        if sampler is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = F.relu(self.bn2(self.conv2(out, p=sampler())))
            out = self.bn3(self.conv3(out, p=sampler()))
            out += self.shortcut(x, sampler=sampler)
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, od, p_s, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.od = od
        self.in_planes = 64

        self.conv1 = create_conv_layer(
            od, True, 3, 64, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = create_bn_layer(od, p_s, 64)
        self.layer1 = self._make_layer(
            od, p_s, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            od, p_s, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            od, p_s, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            od, p_s, block, 512, num_blocks[3], stride=2)
        self.linear = create_linear_layer(
            od, False, 512*block.expansion, num_classes)

    def _make_layer(self, od, p_s, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                od, p_s, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithSampler(*layers)

    def forward(self, x, sampler=None):
        if self.od:
            if sampler is None:
                sampler = BaseSampler(self)
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.layer1(out, sampler=sampler)
            out = self.layer2(out, sampler=sampler)
            out = self.layer3(out, sampler=sampler)
            out = self.layer4(out, sampler=sampler)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(od=False, p_s=[1.]):
    return ResNet(od, p_s, BasicBlock, [2, 2, 2, 2])


def ResNet34(od=False, p_s=[1.]):
    return ResNet(od, p_s, BasicBlock, [3, 4, 6, 3])


def ResNet50(od=False, p_s=[1.]):
    return ResNet(od, p_s, Bottleneck, [3, 4, 6, 3])


def ResNet101(od=False, p_s=[1.]):
    return ResNet(od, p_s, Bottleneck, [3, 4, 23, 3])


def ResNet152(od=False, p_s=[1.]):
    return ResNet(od, p_s, Bottleneck, [3, 8, 36, 3])

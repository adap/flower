import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, make_loss


class DSConv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(input_size, input_size, kernel_size=kernel_size, padding=padding, groups=input_size,
                                   bias=bias)
        self.pointwise = nn.Conv2d(input_size, output_size, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DSCNN(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [DSConv2D(data_shape[0], hidden_size[0], 3, 1),
                  nn.BatchNorm2d(hidden_size[0]),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([DSConv2D(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           nn.BatchNorm2d(hidden_size[i + 1]),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten()])
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(hidden_size[-1], target_size)

    def f(self, x):
        x = self.blocks(x)
        x = self.linear(x)
        return x

    def forward(self, input):
        output = {}
        if 'data' in input:
            output['target'] = self.f(input['data'])
        if 'aug_data' in input:
            output['aug_target'] = self.f(input['aug_data'])
        if 'mix_data' in input:
            output['mix_target'] = self.f(input['mix_data'])
        output['loss'] = make_loss(output, input)
        return output


def dscnn():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['dscnn']['hidden_size']
    model = DSCNN(data_shape, hidden_size, target_size)
    model.apply(init_param)
    return model

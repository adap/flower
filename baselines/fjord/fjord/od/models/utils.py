"""Utility functions for models."""
from torch import nn

from ..layers import ODBatchNorm2d, ODConv2d, ODLinear


def create_linear_layer(od, is_od, *args, **kwargs):
    """Create linear layer.

    :param od: whether to create OD layer
    :param is_od: whether to create OD layer
    :param args: arguments for nn.Linear
    :param kwargs: keyword arguments for nn.Linear
    :return: nn.Linear or ODLinear
    """
    if od:
        return ODLinear(is_od, *args, **kwargs)

    return nn.Linear(*args, **kwargs)


def create_conv_layer(od, is_od, *args, **kwargs):
    """Create conv layer.

    :param od: whether to create OD layer
    :param is_od: whether to create OD layer
    :param args: arguments for nn.Conv2d
    :param kwargs: keyword arguments for nn.Conv2d
    :return: nn.Conv2d or ODConv2d
    """
    if od:
        return ODConv2d(is_od, *args, **kwargs)

    return nn.Conv2d(*args, **kwargs)


def create_bn_layer(od, p_s, *args, **kwargs):
    """Create batch norm layer.

    :param od: whether to create OD layer
    :param p_s: list of p-values
    :param args: arguments for nn.BatchNorm2d
    :param kwargs: keyword arguments for nn.BatchNorm2d
    :return: nn.BatchNorm2d or ODBatchNorm2d
    """
    if od:
        return ODBatchNorm2d(p_s, *args, **kwargs)

    return nn.BatchNorm2d(*args, **kwargs)


class SequentialWithSampler(nn.Sequential):
    """Implements sequential model with sampler."""

    def forward(self, x, sampler=None):
        """Forward method for custom Sequential.

        :param x: input
        :param sampler: the sampler to use.
        :return: Output of sequential
        """
        if sampler is None:
            for module in self:
                x = module(x)
        else:
            for module in self:
                if hasattr(module, "od") and module.od:
                    x = module(x, sampler=sampler)
                elif hasattr(module, "is_od") and module.is_od:
                    x = module(x, p=sampler())
                else:
                    x = module(x)
        return x

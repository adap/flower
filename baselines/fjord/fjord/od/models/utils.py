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
        return ODLinear(*args, is_od=is_od, **kwargs)

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
        return ODConv2d(*args, is_od=is_od, **kwargs)

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
        num_features = kwargs["num_features"]
        del kwargs["num_features"]
        return ODBatchNorm2d(*args, p_s=p_s, num_features=num_features, **kwargs)

    return nn.BatchNorm2d(*args, **kwargs)


class SequentialWithSampler(nn.Sequential):
    """Implements sequential model with sampler."""

    def forward(
        self, input, sampler=None
    ):  # pylint: disable=redefined-builtin, arguments-differ
        """Forward method for custom Sequential.

        :param input: input
        :param sampler: the sampler to use.
        :return: Output of sequential
        """
        if sampler is None:
            for module in self:
                input = module(input)
        else:
            for module in self:
                if hasattr(module, "od") and module.od:
                    input = module(input, sampler=sampler)
                elif hasattr(module, "is_od") and module.is_od:
                    input = module(input, p=sampler())
                else:
                    input = module(input)
        return input

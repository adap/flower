from typing import Tuple, Union

from torch.nn import Module


def check_layer(layer: Module, p: Union[Tuple[Module, float], float]
                ) -> float:
    """
    Check if layer is valid and return p.

    Args:
        layer: PyTorch layer
        p: Ordered dropout p
    """
    # if p is tuple, check layer validity
    if isinstance(p, tuple):
        p, sampled_layer = p
        assert layer == sampled_layer, "Layer mismatch"
    return p

"""Utils function for Ordered Dropout layers."""

from typing import Optional, Tuple, Union

from torch.nn import Module


def check_layer(
    layer: Module, p: Union[Tuple[Module, Optional[float]], Optional[float]]
) -> Optional[float]:
    """Check if layer is valid and return p.

    Args:
        layer: PyTorch layer
        p: Ordered dropout p
    """
    # if p is tuple, check layer validity
    if isinstance(p, tuple):
        p_, sampled_layer = p
        assert layer == sampled_layer, "Layer mismatch"
    else:
        p_ = p

    return p_

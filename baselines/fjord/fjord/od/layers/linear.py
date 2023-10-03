from typing import Union, Tuple

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from .utils import check_layer

__all__ = ["ODLinear"]


class ODLinear(nn.Linear):
    def __init__(self, is_od: bool = True, *args, **kwargs
                 ) -> None:
        super(ODLinear, self).__init__(*args, **kwargs)
        self.is_od = is_od
        self.width = self.out_features
        self.last_input_dim = None
        self.last_output_dim = None

    def forward(self, x: Tensor,
                p: Union[Tuple[Module, float], float] = None) -> Tensor:
        p = check_layer(self, p)
        if not self.is_od and p is not None:
            raise ValueError("p must be None if is_od is False")
        in_dim = x.size(1)  # second dimension is input dimension
        self.last_input_dim = in_dim
        if not p:  # i.e., don't apply OD
            out_dim = self.width
        else:
            out_dim = int(np.ceil(self.width * p))
        self.last_output_dim = out_dim
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(x, weights_red, bias_red)

    def get_slice(self, in_dim: int, out_dim: int) -> Tuple[Tensor, Tensor]:
        weight_slice = self.weight[:in_dim, :out_dim]
        bias_slice = self.bias[:out_dim] if self.bias is not None else None
        return weight_slice, bias_slice

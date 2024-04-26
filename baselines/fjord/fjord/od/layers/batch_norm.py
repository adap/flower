"""BatchNorm using Ordered Dropout."""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["ODBatchNorm2d"]


class ODBatchNorm2d(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Ordered Dropout BatchNorm2d."""

    def __init__(
        self,
        *args,
        p_s: List[float],
        num_features: int,
        affine: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.p_s = p_s
        self.is_od = False  # no sampling is happening here
        self.num_features = num_features
        self.num_features_s = [int(np.ceil(num_features * p)) for p in p_s]
        self.p_to_num_features = dict(zip(p_s, self.num_features_s))
        self.width = np.max(self.num_features_s)
        self.last_input_dim = None

        self.bn = nn.ModuleDict(
            {
                str(num_features): nn.BatchNorm2d(
                    num_features, *args, **kwargs, affine=False
                )
                for num_features in self.num_features_s
            }
        )

        # single track_running_stats
        if affine:
            self.affine = True
            self.weight = nn.Parameter(torch.Tensor(self.width, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(self.width, 1, 1))

        self.reset_parameters()

        # get p into the layer
        for m, p in zip(self.bn, self.p_s):
            self.bn[m].p = p
            self.bn[m].num_batches_tracked = torch.tensor(1, dtype=torch.long)

    def reset_parameters(self):
        """Reset parameters."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        for m in self.bn:
            self.bn[m].reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
        :param x: Input tensor.
        :return: Output of forward pass.
        """
        in_dim = x.size(1)  # second dimension is input dimension
        assert (
            in_dim in self.num_features_s
        ), "input dimension not in selected num_features_s"
        out = self.bn[str(in_dim)](x)
        if self.affine:
            out = out * self.weight[:in_dim] + self.bias[:in_dim]
        return out

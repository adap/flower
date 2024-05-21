"""Ordered Dropout stochastic sampler."""

from collections.abc import Generator
from typing import List

import numpy as np

from .base_sampler import BaseSampler


class ODSampler(BaseSampler):
    """Implements OD sampling per layer up to p-max value.

    :param p_s: list of p-values
    :param max_p: maximum p-value
    """

    def __init__(self, p_s: List[float], max_p: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p_s = np.array([p for p in p_s if p <= max_p])
        self.max_p = max_p

    def width_sampler(self) -> Generator:
        """Sample width."""
        while True:
            p = np.random.choice(self.p_s)
            for _ in range(self.num_od_layers):
                yield p

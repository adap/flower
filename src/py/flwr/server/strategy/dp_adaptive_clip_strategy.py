"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: https://arxiv.org/pdf/1710.06963.pdf
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy 
from flwr.server.strategy.dp_fixed_clip_strategy import DPFixedClipStrategy

import math

class DPAdaptiveClipStrategy(DPFixedClipStrategy):
    """Wrapper for configuring a Strategy for DP with Adaptive Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        total_clients: int,
        noise_multiplier: float = 1,
        init_clip_norm: float = 0.1,
        clip_norm_lr = 0.2,
        clip_norm_target_quantile = 0.5,
        clip_norm_bit_stddev = None
    ) -> None:

        super().__init__(strategy, total_clients, noise_multiplier, init_clip_norm)
        self.num_sampled_clients = math.ceil(strategy.fraction_fit*total_clients)
        self.noise_multiplier = noise_multiplier
        self.clip_norm_lr = clip_norm_lr
        self.clip_norm_target_quantile = clip_norm_target_quantile
        if not clip_norm_bit_stddev:
            self.clip_norm_bit_stddev = self.num_sampled_clients/20.0

    def __repr__(self) -> str:
        rep = f"Strategy with DP with Adaptive Clipping enabled."
        return rep

    def __update_adaptive_parameters(self, results: List[Tuple[ClientProxy, FitRes]]):
        norm_bit_set_count = 0
        for _, fit_res in results:
            if fit_res.metrics["norm_bit"]:
                norm_bit_set_count += 1
        noised_norm_bit_set_count = norm_bit_set_count + np.random.normal(0, self.clip_norm_bit_stddev)
        noised_norm_bit_set_fraction = noised_norm_bit_set_count/len(results)
        self.clip_norm *= math.exp(-self.clip_norm_lr*(noised_norm_bit_set_fraction-self.clip_norm_target_quantile))
        self.noise_std_dev = self.noise_multiplier*self.clip_norm/(self.num_sampled_clients**(-0.5))

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        new_global_model = super().aggregate_fit(rnd, results, failures)
        self.__update_adaptive_parameters(results)
        return new_global_model
    
    
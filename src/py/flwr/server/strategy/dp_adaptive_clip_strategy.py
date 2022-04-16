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
        clip_count_stddev = None
    ) -> None:
        
        super().__init__(strategy, total_clients, noise_multiplier, init_clip_norm)
        self.clip_norm_lr = clip_norm_lr
        self.clip_norm_target_quantile = clip_norm_target_quantile
        if not clip_count_stddev:
            self.clip_count_stddev = self.num_sampled_clients/20.0 if noise_multiplier else 0
        else:
            self.clip_count_stddev = clip_count_stddev
        if noise_multiplier:
            self.noise_multiplier = (self.noise_multiplier**(-2) - (2*self.clip_count_stddev)**(-2))**(-0.5)


    def __repr__(self) -> str:
        rep = f"Strategy with DP with Adaptive Clipping enabled."
        return rep

    def __update_clip_norm(self, results: List[Tuple[ClientProxy, FitRes]]):
        # Calculating number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for _, fit_res in results:
            if fit_res.metrics["norm_bit"]:
                norm_bit_set_count += 1
        # Noising the count       
        noised_norm_bit_set_count = norm_bit_set_count + np.random.normal(0, self.clip_count_stddev)
        noised_norm_bit_set_fraction = noised_norm_bit_set_count/len(results)
        # Geometric update
        self.clip_norm *= math.exp(-self.clip_norm_lr*(noised_norm_bit_set_fraction-self.clip_norm_target_quantile))
        
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        new_global_model = super().aggregate_fit(rnd, results, failures)
        self.__update_clip_norm(results)
        return new_global_model
    
    
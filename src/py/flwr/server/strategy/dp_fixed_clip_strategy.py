"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: https://arxiv.org/pdf/1710.06963.pdf
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

import math
class DPFixedClipStrategy(Strategy):
    """Wrapper for configuring a Strategy for DP with Fixed Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        total_clients: int,
        noise_multiplier: float = 1,
        clip_norm:float = 0.1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.num_sampled_clients = math.ceil(strategy.fraction_fit*total_clients)
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def __repr__(self) -> str:
        rep = f"Strategy with DP with Fixed Clipping enabled."
        return rep

    # Instead of adding noise with std dev sigma = z*clip_norm at server, 
    # add noise with std dev sigma_dash = z*clip_norm at server/sqrt(m) at 
    # each of the m chosen clients.
    def __calc_client_noise_stddev(self):
        return self.noise_multiplier*self.clip_norm/(self.num_sampled_clients**(0.5))

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)



    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        client_instructions = self.strategy.configure_fit(rnd, parameters, client_manager)
        
        for _, fit_ins in client_instructions:
            fit_ins.config["clip_norm"] = self.clip_norm
            fit_ins.config["noise_stddev"] = self.__calc_client_noise_stddev()

        return client_instructions
    
    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(rnd, parameters, client_manager)

    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        for _, fit_res in results:
            fit_res.num_examples = 1
        
        return self.strategy.aggregate_fit(rnd, results, failures)
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(rnd, results, failures)


    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(parameters)
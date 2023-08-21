"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedDF(FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]], 
            failures: List[Tuple[ClientProxy, FitRes],BaseException]
            ) -> Tuple[Parameters ,None, Dict[str, Scalar]]:
        
        return super().aggregate_fit(server_round, results, failures)
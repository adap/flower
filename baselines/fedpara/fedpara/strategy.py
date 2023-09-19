"""FedPara strategy."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fedpara.utils import get_parameters


class FedPara(FedAvg):
    """FedPara strategy."""

    def __init__(
        self,
        algorithm: str,
        net_glob: torch.nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.net_glob = net_glob
        self.w_vec_estimate = np.zeros(
            parameters_to_vector(self.net_glob.parameters()).numel()
        )

    def __repr__(self) -> str:
        """Return the name of the strategy."""
        return self.algorithm

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using FedPara."""

        return ndarrays_to_parameters(get_parameters(self.net_glob)), {}

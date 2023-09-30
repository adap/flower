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
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def __repr__(self) -> str:
        """Return the name of the strategy."""
        return self.algorithm

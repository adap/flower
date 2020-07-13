# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FEDERATED OPTIMIZATION IN HETEROGENEOUS NETWORKS [Li et al., 2020] strategy.

Paper: https://proceedings.mlsys.org/static/paper_files/mlsys/2020/176-Paper.pdf
"""


import random
from typing import Callable, Dict, List, Optional, Tuple

from flwr.client_manager import ClientManager
from flwr.client_proxy import ClientProxy
from flwr.typing import FitIns, Weights

from .fedavg import FedAvg
from .parameter import weights_to_parameters


class FedProxMu0(FedAvg):
    """FedProx strategy implementation with mu = 0."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = True,
        e_min: int = 1,
        e_max: int = 5,
    ) -> None:
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.e_min = e_min
        self.e_max = e_max
        self.epochs: Dict[str, int] = {}

    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate model weights using an evaluation function (if provided)."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        return self.eval_fn(weights)

    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Configure clients w/ sampled values for E
        parameters = weights_to_parameters(weights)
        client_fit_configs = []
        for client in clients:
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(rnd)
                # Ignore timeout
                config.pop("timeout")

            # Sample E
            config["epochs"] = str(self._get_epochs(client.cid))

            # Create fit instructions
            fit_ins = (parameters, config)
            client_fit_configs.append((client, fit_ins))

        # Return client/config pairs
        return client_fit_configs


    def _get_epochs(self, cid: str) -> int:
        if cid not in self.epochs.keys():
            self.epochs[cid] = random.randint(self.e_min, self.e_max)
        return self.epochs[cid]

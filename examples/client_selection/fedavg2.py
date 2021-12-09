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
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""

import numpy

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    Weights,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.typing import PropertiesIns


class FedAvg2(FedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
        )

    def __repr__(self) -> str:
        rep = f"FedAvg2(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # Block until at least num_clients are connected.
        client_manager.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(client_manager.clients)
        ins = PropertiesIns(config={})
        ins.config = {}
        num_samples_vect = [
            client_manager.clients[cid]
            .get_properties(ins=ins)
            .properties["num_samples"]
            for cid in available_cids
        ]
        prob = numpy.array(num_samples_vect) / numpy.sum(num_samples_vect)
        sampled_cids = numpy.random.choice(
            available_cids, size=sample_size, replace=False, p=prob
        )
        clients = [client_manager.clients[cid] for cid in sampled_cids]

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

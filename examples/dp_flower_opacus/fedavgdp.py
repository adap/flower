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
"""Federated Averaging DP strategy."""

from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

import flwr.server.server as server
from flwr.common import FitRes, Parameters, Scalar, Weights
from flwr.server.strategy import FedAvg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedAvgDp(FedAvg):
    """This class implements the FedAvg strategy for Differential Privacy context."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        FedAvg.__init__(
            self,
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )

        # Keep track of the maximum possible privacy budget
        self.max_epsilon = 0.0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple["ClientProxy", FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

        if not results:
            return None

        # Get the privacy budget of each client
        accepted_results = []
        epsilons = []
        for c, r in results:
            # Check if client can be accepted or not
            if r.metrics["accept"]:
                accepted_results.append([c, r])
                epsilons.append(r.metrics["epsilon"])
            else:
                # Disconnect any client whose privacy budget was exceeded.
                server.reconnect_client(c, server.Reconnect(None))

        results = accepted_results
        if epsilons:
            self.max_epsilon = max(self.max_epsilon, max(epsilons))

        logger.info("Privacy budget ε at round {}: {:.3f}", rnd, self.max_epsilon)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_fit(rnd, results, failures)

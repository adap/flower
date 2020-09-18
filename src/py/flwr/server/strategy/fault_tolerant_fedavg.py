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
"""Fault-tolerant variant of FedAvg strategy."""


from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import EvaluateRes, FitRes, Weights, parameters_to_weights
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .fedavg import FedAvg


class FaultTolerantFedAvg(FedAvg):
    """Configurable fault-tolerant FedAvg strategy implementation."""

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
        min_completion_rate_fit: float = 0.5,
        min_completion_rate_evaluate: float = 0.5,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=True,
        )
        self.completion_rate_fit = min_completion_rate_fit
        self.completion_rate_evaluate = min_completion_rate_evaluate

    def __repr__(self) -> str:
        return "FaultTolerantFedAvg()"

    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None
        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.completion_rate_fit:
            # Not enough results for aggregation
            return None
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return aggregate(weights_results)

    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.completion_rate_evaluate:
            # Not enough results for aggregation
            return None
        return weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss, evaluate_res.accuracy)
                for client, evaluate_res in results
            ]
        )

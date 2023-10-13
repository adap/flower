# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Federated XGBoost [Ma et al., 2023] strategy.

Strategy in the horizontal setting based on building Neural Network and averaging on
prediction outcomes.

Paper: Coming
"""


from logging import WARNING
from typing import Any, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate
from .fedavg import FedAvg


class FedXgbNnAvg(FedAvg):
    """Configurable FedXgbNnAvg strategy implementation."""

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedXgbNnAvg(accept_failures={self.accept_failures})"
        return rep

    def evaluate(
        self, server_round: int, parameters: Any
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        eval_res = self.evaluate_fn(server_round, parameters, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Any], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters[0].parameters),  # type: ignore # noqa: E501 # pylint: disable=line-too-long
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate XGBoost trees from all clients
        trees_aggregated = [fit_res.parameters[1] for _, fit_res in results]  # type: ignore # noqa: E501 # pylint: disable=line-too-long

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return [parameters_aggregated, trees_aggregated], metrics_aggregated

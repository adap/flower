# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""FedAvg with in-place aggregation."""


from functools import reduce
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


class FedAvgInPlace(FedAvg):
    """FedAvg with in-place aggregation."""

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvgInPlace(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Count total examples
        num_examples_total = sum([fit_res.num_examples for _, fit_res in results])

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        # Let's do in-place aggregation
        # get first result, then add up each other
        params = [
            scaling_factors[0] * x
            for x in parameters_to_ndarrays(results[0][1].parameters)
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [
                reduce(np.add, layer_updates) for layer_updates in zip(params, res)
            ]

        # convert to parameters
        parameters_aggregated = ndarrays_to_parameters(params)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

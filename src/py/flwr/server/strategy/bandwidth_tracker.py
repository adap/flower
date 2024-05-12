# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Bandwidth Tracker Strategy."""


from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class BandwidthTracker(Strategy):
    """Strategy wrapper for tracking the bandwidth during FL rounds.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which to track its incoming and outgoing bandwidth.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
    ) -> None:
        super().__init__()

        self.strategy = strategy

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Bandwidth Tracker Strategy Wrapper"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        params = parameters_to_ndarrays(parameters)
        size_in_bytes = (array.nbytes for array in params)
        log(INFO, f"Broadcasting model size: {size_in_bytes} Bytes")

        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Track the receving bandwidth."""
        if failures:
            return None, {}

        for _, res in results:
            params = parameters_to_ndarrays(res.parameters)
            size_in_bytes = (array.nbytes for array in params)
            log(INFO, f"Receiving model size: {size_in_bytes} Bytes")

        parameters_aggregated, metrics_aggregated = self.aggregate_fit(
            server_round, results, failures
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)

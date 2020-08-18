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
"""Flower server strategy."""


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Weights
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class Strategy(ABC):
    """Abstract class to implement custom server strategies."""

    @abstractmethod
    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Arguments:
            rnd: Integer. The current round of federated learning.
            weights: Weights. The current (global) model weights.
            client_manager: ClientManager. The client manger which knows about all
                currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy` is not
            included in this list, it means that this `ClientProxy` will not participate
            in the next round of federated learning.
        """

    @abstractmethod
    def on_configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

    @abstractmethod
    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate training results."""

    @abstractmethod
    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results."""

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate the current model weights."""

    @abstractmethod
    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Conclude federated learning round and decide whether to continue or
        not."""

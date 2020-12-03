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
    """Abstract base class for server strategy implementations."""

    @abstractmethod
    def configure_fit(
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
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """

    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate training results.

        Arguments:
            rnd: int. The current round of federated learning.
            results: List[Tuple[ClientProxy, FitRes]]. Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: List[BaseException]. Exceptions that occurred while the server
                was waiting for client updates.

        Returns:
            Optional `flwr.common.Weights`. If weights are returned, then the server
            will treat these as the new global model weights (i.e., it will replace the
            previous weights with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable results)
            then the server will no update the previous model weights, the updates
            received in this round are discarded, and the global model weights remain
            the same.
        """

    @abstractmethod
    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Arguments:
            rnd: Integer. The current round of federated learning.
            weights: Weights. The current (global) model weights.
            client_manager: ClientManager. The client manger which knows about all
                currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results.

        Arguments:
            rnd: int. The current round of federated learning.
            results: List[Tuple[ClientProxy, FitRes]]. Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: List[BaseException]. Exceptions that occurred while the server
                was waiting for client updates.

        Returns:
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Evaluate the current model weights.

        This function can be used to perform centralized (i.e., server-side) evaluation
        of model weights.

        Arguments:
            weights: Weights. The current (global) model weights.

        Returns:
            The evaluation result, usually a Tuple containing loss and accuracy.
        """

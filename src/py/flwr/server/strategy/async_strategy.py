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
"""Flower server async strategy."""


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class AsyncStrategy(Strategy, ABC):
    """Abstract base class for async server strategy implementations."""

    @abstractmethod
    def async_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        results_cids: List[str],
        failures_cids: List[str],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """


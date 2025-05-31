# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower server client results strategy to detect malicious actors."""


from abc import ABC, abstractmethod
from typing import Union

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy


class ClientResultsStrategy(ABC):
    """Abstract base class for server client results strategy implementations."""

    @abstractmethod
    def validate_client_results(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[
        list[tuple[ClientProxy, FitRes]],
        list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ]:
        """Scan client training results for malicious activity.

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
        trusted_results : Tuple[
            List[tuple[ClientProxy, FitRes]],
            List[Union[tuple[ClientProxy, FitRes], BaseException]
        ]
            The tuple represents the results that should be used for the next evaluation
            of the model training.
        """

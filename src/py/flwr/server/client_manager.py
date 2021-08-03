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
"""Flower ClientManager."""


import random
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .client_proxy import ClientProxy
from .criterion import Criterion


class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients.
        
        Returns
        -------
        int
            Number of currently available clients.
        """

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance with ClientManager.

        Parameters
        ----------
        client : ClientProxy
            Client to be registered with ClientManager

        Returns
        -------
        bool
            Indication that registration was successful. False if ClientProxy
            is already registered or can not be registered for any reason.
        """

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.
        This method is idempotent.

        Parameters
        ----------
        client : ClientProxy
            Client to be unregistered with ClientManager
        """

    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances and return them.
        
        Parameters
        ----------
        num_clients : int
            Number of clients to sample
        min_num_clients : int (default: None)
            Optional minimum number of clients to be available before sampling.
            If the minimum number is not yet available block until than.
        criterion : Criterion (default: None)
            Criterion object which to allow criterion sampling
        """


class SimpleClientManager(ClientManager):
    """Default implementation of the ClientManager interface."""

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.clients)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached.

        Parameters
        ----------
        num_clients : int
            Number of clients to wait for and block until they are available.
        timeout : int (default: 86400)
            A timeout in seconds after giving the maximum time to
            wait and after which `False` is returned.

        Returns
        -------
        result : bool
            Indicates that the required number of clients is present. 
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def num_available(self) -> int:
        """Return the number of available clients.
        
        Returns
        -------
        int
            Number of currently available clients.
        """
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance with ClientManager.

        Parameters
        ----------
        client : ClientProxy
            Client to be registered with ClientManager

        Returns
        -------
        bool
            Indication that registration was successful. False if ClientProxy
            is already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.
        This method is idempotent.

        Parameters
        ----------
        client : ClientProxy
            Client to be unregistered with ClientManager

        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

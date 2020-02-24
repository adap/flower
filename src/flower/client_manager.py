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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .client import Client
from .criterion import Criterion


class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def register(self, client: Client) -> bool:
        """Register Flower Client instance.

        Returns:
            bool: Indicating if registration was successful
        """
        raise NotImplementedError()

    @abstractmethod
    def unregister(self, client: Client) -> None:
        """Unregister Flower Client instance."""
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self, num_clients: int, criterion: Optional[Criterion] = None
    ) -> List[Client]:
        """Sample a number of Flower Client instances."""
        raise NotImplementedError()


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients: Dict[str, Client] = {}

    def __len__(self):
        return len(self.clients)

    def register(self, client: Client) -> bool:
        """Register Flower Client instance.

        Returns:
            bool: Indicating if registration was successful. False if client is already
                registered or can not be registered for any reason
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        return True

    def unregister(self, client: Client) -> None:
        """Unregister Flower Client instance.

        This method is idempotent.
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

    def sample(
        self, num_clients: int, criterion: Optional[Criterion] = None
    ) -> List[Client]:
        """Sample a number of Flower Client instances."""
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

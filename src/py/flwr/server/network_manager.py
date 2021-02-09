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

import threading
from flwr.client.keras_client import KerasClient, KerasClientWrapper

import grpc
from abc import ABC, abstractmethod
from typing import Optional, List

from flwr.client.numpy_client import NumPyClient, NumPyClientWrapper
from flwr.server.client_manager import ClientManager
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server
from flwr.server.in_memory_server.in_memory_client_proxy import InMemoryClientProxy
from flwr.client import Client


class NetworkManager(ABC):
    """Abstract base class for managing networks."""

    def __init__(self, client_manager: ClientManager) -> None:
        """Init method"""
        self.client_manager = client_manager

    @abstractmethod
    def start(self) -> None:
        """Start the network."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the network."""


class GRPCNetworkManager(NetworkManager):
    """Provides a gRPC Network."""

    def __init__(
        self,
        client_manager: ClientManager,
        server_address: str,
        grpc_max_message_length: int,
    ) -> None:
        super().__init__(client_manager)

        self.server_address = server_address
        self.grpc_max_message_length = grpc_max_message_length

        self.server: Optional[grpc.Server] = None

    def start(self):
        self.server = start_insecure_grpc_server(
            client_manager=self.client_manager,
            server_address=self.server_address,
            max_message_length=self.grpc_max_message_length,
        )

    def stop(self):
        self.server.stop(1)


class SimpleInMemoryNetworkManager(NetworkManager):
    def __init__(
        self, client_manager: ClientManager, clients: List[Client], parallel: int = 1
    ) -> None:
        super().__init__(client_manager)

        # Use a Semaphore for parallelism. Warning this can create issues with
        # TensorFlow if the global state TF influences the results each client produces.
        # Take care of that or don't increase parallism
        self._cv = threading.Semaphore(value=parallel)

        Wrapper = None

        if len(clients) > 0:
            if isinstance(clients[0], NumPyClient):
                Wrapper = NumPyClientWrapper
            elif isinstance(clients[0], KerasClient):
                Wrapper = KerasClientWrapper
            else:
                raise Exception("Client Class is not yet supported.")

        self.client_proxies = [
            InMemoryClientProxy(cid=index, client=Wrapper(client), lock=self._cv)
            for index, client in enumerate(clients)
        ]

    def start(self) -> None:
        for client_proxy in self.client_proxies:
            self.client_manager.register(client_proxy)

    def stop(self) -> None:
        for client_proxy in self.client_proxies:
            self.client_manager.unregister(client_proxy)

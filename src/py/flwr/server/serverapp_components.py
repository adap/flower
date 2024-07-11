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
"""ServerAppComponents for the ServerApp."""


from dataclasses import dataclass
from typing import Optional

from .client_manager import ClientManager
from .server import Server
from .server_config import ServerConfig
from .strategy import Strategy


@dataclass
class ServerAppComponents:
    """Components to construct a ServerApp.

    Parameters
    ----------
    server : Optional[Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, one will be created internally.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[Strategy] (default: None)
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `flwr.server.strategy.FedAvg` will be used.
    client_manager : Optional[ClientManager] (default: None)
        An implementation of the class `flwr.server.ClientManager`. If no
        implementation is provided, then `flwr.server.SimpleClientManager`
        will be used.
    """

    server: Optional[Server]
    config: Optional[ServerConfig]
    strategy: Optional[Strategy]
    client_manager: Optional[ClientManager]

    def __init__(
        self,
        server: Optional[Server] = None,
        config: Optional[ServerConfig] = None,
        strategy: Optional[Strategy] = None,
        client_manager: Optional[ClientManager] = None,
    ) -> None:
        self._server = server
        self._config = config
        self._strategy = strategy
        self._client_manager = client_manager

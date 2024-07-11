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
from typing import Dict, Optional

from .server import Server, ServerConfig
from .strategy import Strategy
from .client_manager import ClientManager

@dataclass
class ServerAppComponents:
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
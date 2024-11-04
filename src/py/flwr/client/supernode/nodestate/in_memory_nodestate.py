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
"""In-memory NodeState implementation."""

import threading
from typing import Optional

from flwr.client.clientapp.clientappio_servicer import ClientAppInputs, ClientAppOutputs
from flwr.client.supernode.nodestate import NodeState


class InMemoryNodeState(NodeState):
    """In-memory NodeState implementation."""

    def __init__(self) -> None:

        # Map token to clientapp_inputs/outputs
        self.clientapp_inputs: dict[int, ClientAppInputs] = {}
        self.clientapp_outputs: dict[int, ClientAppOutputs] = {}

        self.lock = threading.Lock()

    def set_clientapp_inputs(
        self, token: int, clientapp_inputs: ClientAppInputs
    ) -> None:
        """Set the ClientAppInputs for the specified `token`."""
        self.clientapp_inputs[token] = clientapp_inputs

    def get_clientapp_inputs(self, token: int) -> Optional[ClientAppInputs]:
        """Get the ClientAppInputs for the specified `token`."""
        return self.clientapp_inputs.get(token)

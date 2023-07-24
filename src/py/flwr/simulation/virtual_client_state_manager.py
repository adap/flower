# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Flower Client State Manager for virtual/simulated clients."""

from logging import INFO
from typing import Any, Dict

from flwr.client import ClientState
from flwr.common.logger import log


class VirtualClientStateManager:
    def __init__(self, init_from_file: str = None):
        self._states: Dict[str, ClientState] = {}
        if init_from_file is not None:
            self._load_states_from_file(init_from_file)

    def track_state(self, client_key: str, client_state: ClientState):
        """The state of a new (virtual) client is to be tracked."""

        # only add those clients that aren't present in the dictionary
        # client's state might be present already if the state manager
        # loaded client states from a file during __init__()
        if client_key not in self._states.keys():
            self._states[client_key] = client_state

    def get_client_state(self, client_key: str):
        return self._states[client_key].fetch()

    def update_client_state(self, client_key: str, client_state_data: Dict[str, Any]):
        self._states[client_key].update(client_state_data)

    def _load_states_from_file(self, states_file: str):
        """Load state for all/some clients from a file."""
        log(INFO, f"Initializing state from: {states_file}")
        raise NotImplementedError()


class SimpleVirtualClientStateManager(VirtualClientStateManager):
    def __init__(self):
        super().__init__()

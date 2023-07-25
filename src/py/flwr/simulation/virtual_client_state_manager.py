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

from flwr.client import ClientState, InFileSystemClientState, InMemoryClientState
from flwr.common.logger import log


class VirtualClientStateManager:
    def __init__(self, init_from_file: str = None, verbose: bool = False):
        self._states: Dict[str, ClientState] = {}
        self.verbose = verbose
        if init_from_file is not None:
            self._load_states_from_file(init_from_file)

    def track_state(self, client_key: str, client_state: ClientState):
        """The state of a new (virtual) client is to be tracked."""

        # only add those clients that aren't present in the dictionary
        # client's state might be present already if the state manager
        # loaded client states from a file during __init__()
        if self.verbose:
            log(INFO, f"Tracking state for client: {client_key}")
        if client_key not in self._states.keys():
            self._states[client_key] = client_state

    def get_client_state(self, client_key: str, **kwargs):
        if self.verbose:
            log(INFO, f"Fetching state for client: {client_key}")
        return self._states[client_key].fetch_state(**kwargs)

    def update_client_state(
        self, client_key: str, client_state_data: Dict[str, Any], **kwargs
    ):
        if self.verbose:
            log(INFO, f"Updating state for client: {client_key} -> {client_state_data}")
        self._states[client_key].update_state(client_state_data, **kwargs)

    def _load_states_from_file(self, states_file: str):
        """Load state for all/some clients from a file."""
        if self.verbose:
            log(INFO, f"Initializing state from: {states_file}")
        raise NotImplementedError()


class SimpleVirtualClientStateManager(VirtualClientStateManager):
    def __init__(self, client_state_type=InMemoryClientState, **kwargs):
        super().__init__(**kwargs)
        self.client_state_type = client_state_type

    def track_state(self, client_key: str):
        """Track state of a virtual client.

        By default, virtual clients use InMemoryClientState."""
        super().track_state(client_key, self.client_state_type())

        # setup state (does nothing)
        self._states[client_key].setup_state()


class InFileSystemVirtualClientStateManager(VirtualClientStateManager):
    def __init__(
        self,
        state_dir: str = "client_states",
        load_state_from_disk_at_init: bool = False,
        client_state_type=InFileSystemClientState,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.client_state_type = client_state_type
        # directory where all client states will be stored
        self.clients_state_dir = state_dir
        # if true and if the client state is found in the file system
        # at initialization, then the state will be set by reading from disk
        self.load_state_from_disk_at_init = load_state_from_disk_at_init

    def track_state(self, client_key: str):
        # want each client to have it's own state file. we can differentiate them using the key
        # assigned to each client's proxy (i.e. the client id)
        super().track_state(
            client_key, self.client_state_type(state_filename=f"client_{client_key}")
        )

        # setup state (does nothing)
        self._states[client_key].setup_state(
            state_dir=self.clients_state_dir,
            create_directory=True,
            load_if_exist=self.load_state_from_disk_at_init,
        )

    def update_client_state(self, client_key: str, client_state_data: Dict[str, Any]):
        return super().update_client_state(client_key, client_state_data, to_disk=True)

    def get_client_state(self, client_key: str):
        return super().get_client_state(client_key, from_disk=True)

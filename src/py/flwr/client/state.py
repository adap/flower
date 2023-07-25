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
"""Flower Client State."""

import pickle
from abc import ABC, abstractmethod
from logging import DEBUG, WARNING
from pathlib import Path
from typing import Any, Dict, Optional

from flwr.common.logger import log


class ClientState(ABC):
    """Abstract base class for Flower client state."""

    @abstractmethod
    def setup_state(self) -> None:
        """Initialize client state."""

    @abstractmethod
    def fetch_state(
        self,
        timeout: Optional[float],
    ) -> Dict:
        """Return the client's state."""

    @abstractmethod
    def update_state(
        self,
        state: Dict,
        timeout: Optional[float],
    ) -> None:
        """Update the client's state."""


class InMemoryClientState(ClientState):
    """An in-memory Client State class for clients to record their state and use it
    across rounds.

    This in-memory state is suitable for both "real" and "simulated" clients.
    Note the state won't persist once the Federated Learning workload is completed. If
    you would like to save/load the state to/from disk, use the
    `InFileSystemClientState` class.
    """

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Any] = {}

    def setup_state(self) -> None:
        """Initialise the state."""
        pass

    def fetch_state(self) -> Dict:
        return self.state

    def update_state(self, state: Dict) -> None:
        # TODO: deep copy for peace of mind?
        self.state = state


class InFileSystemClientState(ClientState):
    """A Client State class that enables the loading/recording from/to the file system.

    In this way, the client state can be retrieved/updated across different Federated
    Learning workloads. This client state class can be used to initialize the client
    state at the beginning of the client's life.
    """

    def __init__(
        self,
        state_filename: str = "client_state",
        keep_in_memory: bool = True,
    ):
        super().__init__()
        self.state_filename = state_filename
        self.keep_in_memory = keep_in_memory
        self.path = None  # to be setup upon setup() call
        self.state: Dict[str, Any] = {}

    def setup_state(
        self, state_dir: str, create_directory: bool, load_if_exist: bool = True
    ) -> None:
        """Initialize state by loading it from disk if exists.

        Else, create file directory structure and init an empty state.
        """
        self.path = Path(state_dir)
        if self.path.exists():
            # load state
            if load_if_exist:
                self._load_state()
        else:
            if create_directory:
                log(
                    DEBUG, f"Creating directory for client state: {self.path.resolve()}"
                )
                self.path.mkdir(parents=True)
                self._write_state()
            else:
                log(
                    WARNING,
                    f"A {self.__class__.__name__} is not present in {self.path}."
                    "Either set `create_directory=True` or pass a path that exists.",
                )
                log(
                    WARNING,
                    "This client's state will remain in memory for the duration of the"
                    "experiment.",
                )

    def _load_state(self):
        """Load client state from pickle."""
        state_file = self.path / f"{self.state_filename}.pkl"
        if state_file.exists():
            with open(state_file, "rb") as handle:
                state = pickle.load(handle)

            # update state (but don't write to disk since we
            # just read from it)
            self.update_state(state, to_disk=False)
        else:
            log(WARNING, f"State `{state_file}` does not exist.")

    def _write_state(self) -> bool:
        """Write client state to pickle."""
        state_file = self.path / f"{self.state_filename}.pkl"
        if self.path.exists():
            with open(state_file, "wb") as handle:
                state = self.fetch_state()
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        else:
            log(
                WARNING,
                "State can't be saved to disk (path doesn't exist). The client "
                "state persists in memory for the duration of the experiment.",
            )
            return False

    def fetch_state(self, from_disk: bool = False) -> Dict:
        """Return client state."""
        # you might want to read from disk in case there is another application or
        # service updating some of the elements in the state of this client
        # (e.g. a data source)
        if from_disk:
            # TODO: probably we want to wrap this around a FileLock?
            self._load_state()
        return self.state

    def update_state(self, state: Dict, to_disk: bool = True) -> None:
        """Update client state."""
        self.state = state
        if to_disk:
            # save state to disk
            # TODO: probably we want to wrap this around a FileLock?
            saved = self._write_state()

            # free state after saving it to disk
            if saved and not self.keep_in_memory:
                self.state.clear()

class InFileSystemVirtualClientState(InFileSystemClientState):
    """In-FileSystem Client State for Clients in simulation.
    
    It behaves very much like InFileSystemClientState but, since
    clients do not persist across rounds and might be spawned in
    different machines each time, accessing the filesystem should
    be done at specific times. Concretely, before a client is spawned
    (e.g. to do `fit()`) and when it ends its task. Then, the ClientProxy
    object (which is the entity that interfaces with the server) can
    update the state in the file system. In this way, the client gets
    exposed an InMemoryClientState object at runtime.
    
    BEAR IN MIND this class won't do anything smart if the directory
    of states already exist, and as a result they will likely be overwritten
    each time you run the same simulation. To avoid this, you might want
    to pass `state_dir` at init which takes, for instance, the datetime
    when you run the code.
    """

    def __init__(self, state_dir: str = "client_states",
                 update_fs_each_round: bool = True,
                 load_state_from_disk_at_init: bool=False):
        super().__init__()
        self.update_fs_each_round = update_fs_each_round
        # directory where all client states will be stored
        self.clients_state_dir = state_dir
        # if true and if the client state is found in the file system
        # at initialization, then the state will be set by reading from disk
        self.load_state_from_disk_at_init = load_state_from_disk_at_init
    
    def state_setup(self) -> None:
        return super().state_setup(self.clients_state_dir,
                             create_directory=True,
                             load_if_exist=self.load_state_from_disk_at_init)

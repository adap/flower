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

    @property
    def _state(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self) -> None:
        """Initialize client state."""

    @abstractmethod
    def fetch(
        self,
        timeout: Optional[float],
    ) -> Dict:
        """Return the client's state."""

    @abstractmethod
    def update(
        self,
        state: Dict,
        timeout: Optional[float],
    ) -> None:
        """Update the client's state."""


class InMemoryClientState(ClientState):
    _state: Dict[str, Any] = {}

    # TODO: this type of state is intended for those settings were client want to preserve some of their info across rounds
    # but that do now want to save anything to disk (#!!! not even at the end of the experiment). How do we warn users about this?
    # Should we instead design a flag to save to disk at the end? if so, there is too much overlap between InMemoryClientState and 
    # InFileSystemClientState

    def __init__(self):
        super().__init__()

    def setup(self) -> None:
        self._state = {}

    def fetch(self) -> Dict:
        return self._state

    def update(self, state: Dict) -> None:
        # TODO: deep copy for peace of mind?
        self._state = state


class InFileSystemClientState(ClientState):
    _state: Dict[str, Any] = {}

    def __init__(
        self,
        state_filename: str = "client_state",
        keep_in_memory: bool = True,
    ):
        self.state_filename = state_filename
        self.keep_in_memory = keep_in_memory
        self.path = None # to be setup upon setup() call

    def setup(self, state_dir: str, create_directory: bool) -> None:
        """Initialize state by loading it from disk if exists.

        Else, create file directory structure and init an empty state.
        """
        self.path = Path(state_dir)
        if self.path.exists():
            # load state
            self._load_state()
        else:
            # state is empty
            self._state = {}

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
                # TODO: dynamically revert to InMemoryClientState type?


    def _load_state(self):
        """Load client state from pickle."""
        state_file = self.path / f"{self.state_filename}.pkl"
        if self.path.exists():
            with open(state_file, "rb") as handle:
                state = pickle.load(handle)

        # update state (but don't write to disk since we
        # just read from it)
        self.update(state, to_disk=False)

    def _write_state(self) -> bool:
        """Write client state to pickle."""
        state_file = self.path / f"{self.state_filename}.pkl"
        if self.path.exists():
            with open(state_file, "wb") as handle:
                state = self.fetch()
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        else:
            log(
                WARNING,
                "State can't be saved to disk (path doesn't exist). The client "
                "state persists in memory for the duration of the experiment.",
            )
            return False

    def fetch(self, from_disk: bool = False) -> Dict:
        """Return client state."""
        # you might want to read from disk in case there is another application or
        # service updating some of the elements in the state of this client
        # (e.g. a data source)
        if from_disk:
            # TODO: probably we want to wrap this around a FileLock?
            self._load_state()
        return self._state

    def update(self, state: Dict, to_disk: bool = True) -> None:
        """Update client state."""
        self._state = state
        if to_disk:
            # save state to disk
            # TODO: probably we want to wrap this around a FileLock?
            saved = self._write_state()

            # free state after saving it to disk
            if saved and not self.keep_in_memory:
                self._state.clear()


# if __name__ == "__main__":
#     ss = InMemoryClientState()
#     print(f"{ss.fetch() = }")
#     ss.update({"hello": 123})
#     print(f"{ss.fetch() = }")

#     ssfs = InFileSystemClientState(state_dir="./here/teststate")
#     print(f"{ssfs.fetch() = }")
#     ssfs.update({"hello": 123})
#     ssfs.update({"hello": 1235}, to_disk=False)
#     print(f"{ssfs.fetch() = }")
#     print(f"{ssfs.fetch(from_disk=True) = }")

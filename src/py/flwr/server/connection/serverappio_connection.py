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
"""Flower ServerApp connection."""


from abc import ABC, abstractmethod
from collections.abc import Iterable

from flwr.common import Context, Message
from flwr.common.typing import Fab, Run


class ServerAppIo(ABC):
    """Abstract base class for the ServerAppIo API."""

    @abstractmethod
    def get_node_ids(self) -> Iterable[int]:
        """Get the node IDs of all available SuperNodes."""

    @abstractmethod
    def push_messages(self, messages: Iterable[Message]) -> None:
        """Push messages to the SuperLink."""

    @abstractmethod
    def pull_messages(self, message_ids: Iterable[int]) -> Iterable[Message]:
        """Pull reply messages from the SuperLink."""

    @abstractmethod
    def get_run(self, run_id: int) -> Run:
        """Get run info."""

    @abstractmethod
    def get_fab(self, fab_hash: str) -> Fab:
        """Get FAB file."""

    @abstractmethod
    def pull_inputs(self) -> tuple[Fab, Context]:
        """Pull ServerApp inputs from the SuperLink."""

    @abstractmethod
    def push_outputs(self, run_id: int, context: Context) -> None:
        """Push ServerApp outputs to the SuperLink."""

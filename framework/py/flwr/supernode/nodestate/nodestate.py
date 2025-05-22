# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Abstract base class NodeState."""


from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from flwr.common import Context, Message
from flwr.common.typing import Run


class NodeState(ABC):
    """Abstract base class for node state."""

    @abstractmethod
    def set_node_id(self, node_id: int) -> None:
        """Set the node ID."""

    @abstractmethod
    def get_node_id(self) -> int:
        """Get the node ID."""

    @abstractmethod
    def get_message(
        self,
        *,
        run_id: Optional[int | list[int]] = None,
        is_reply: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Message]:
        """Retrieve messages based on the specified filters.

        If a filter is set to None, it is ignored.
        If a list is provided for a filter, it is treated as an OR condition.

        Parameters
        ----------
        run_id : Optional[int | list[int]] (default: None)
            Run ID or list of run IDs to filter by.
        is_reply : Optional[bool] (default: None)
            If True, filter for reply messages; if False, filter for non-reply
            (instruction) messages.
        limit : Optional[int] (default: None)
            Maximum number of messages to return. If None, no limit is applied.

        Returns
        -------
        dict[str, Message]
            Dictionary of messages matching the filters, keyed by their object ID.

        Notes
        -----
        **IMPORTANT:** Retrieved messages will be **deleted** after retrieval!
        """

    @abstractmethod
    def store_message(self, message: Message, object_id: str) -> None:
        """Store a message.

        Parameters
        ----------
        message : Message
            The message to store.
        object_id : str
            The object ID of the message.
        """

    @abstractmethod
    def store_run(self, run: Run) -> None:
        """Store a run.

        Parameters
        ----------
        run : Run
            The `Run` instance to store.
        """

    @abstractmethod
    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve a run by its ID.

        Parameters
        ----------
        run_id : int
            The ID of the run to retrieve.

        Returns
        -------
        Optional[Run]
            The `Run` instance if found, otherwise None.
        """

    @abstractmethod
    def get_run_ids(self) -> list[int]:
        """Retrieve all stored run IDs.

        Returns
        -------
        list[int]
            List of all stored run IDs.
        """

    @abstractmethod
    def store_context(self, context: Context) -> None:
        """Store a context.

        Parameters
        ----------
        context : Context
            The context to store.
        """

    @abstractmethod
    def get_context(self, run_id: int) -> Optional[Context]:
        """Retrieve a context by its run ID.

        Parameters
        ----------
        run_id : int
            The ID of the run with which the context is associated.

        Returns
        -------
        Optional[Context]
            The `Context` instance if found, otherwise None.
        """

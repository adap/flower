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
"""Abstract base class FederationManager."""


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flwr.common.typing import Federation

if TYPE_CHECKING:
    from flwr.server.superlink.linkstate.linkstate import LinkState


class FederationManager(ABC):
    """Abstract base class for FederationManager."""

    @property
    def linkstate(self) -> "LinkState":
        """Return the LinkState instance."""
        if not (ret := getattr(self, "_linkstate", None)):
            raise RuntimeError("linkstate not set. Assign to linkstate property first.")
        return ret  # type: ignore

    @linkstate.setter
    def linkstate(self, linkstate: "LinkState") -> None:
        """Set the LinkState instance."""
        self._linkstate = linkstate

    @abstractmethod
    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""

    @abstractmethod
    def has_member(self, flwr_aid: str, federation: str) -> bool:
        """Check if the given account is a member of the federation."""

    @abstractmethod
    def filter_nodes(self, node_ids: set[int], federation: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""

    @abstractmethod
    def has_node(self, node_id: int, federation: str) -> bool:
        """Given a node ID, check if it is in the federation."""

    @abstractmethod
    def get_federations(self, flwr_aid: str) -> list[tuple[str, str]]:
        """Get federations (name, description) of which the account is a member."""

    @abstractmethod
    def get_details(self, federation: str) -> Federation:
        """Get details of the federation."""

    @abstractmethod
    def create_federation(
        self, flwr_aid: str, name: str, description: str
    ) -> Federation:
        """Create a new federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account creating the federation.
        name : str
            The unique name of the federation.
        description : str
            A human-readable description of the federation.

        Returns
        -------
        Federation
            The created federation.
        """

    @abstractmethod
    def archive_federation(self, flwr_aid: str, name: str) -> None:
        """Archive an existing federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account archiving the federation.
        name : str
            The name of the federation to archive.
        """

    @abstractmethod
    def add_supernodes(
        self, flwr_aid: str, federation: str, node_ids: set[int]
    ) -> None:
        """Add SuperNodes to a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account adding the SuperNodes.
        federation : str
            The name of the federation.
        node_ids : set[int]
            The IDs of the SuperNodes to add.
        """

    @abstractmethod
    def remove_supernodes(
        self, flwr_aid: str, federation: str, node_ids: set[int]
    ) -> None:
        """Remove SuperNodes from a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account removing the SuperNodes.
        federation : str
            The name of the federation.
        node_ids : set[int]
            The IDs of the SuperNodes to remove.
        """

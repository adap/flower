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

from flwr.app import Message


class FederationManager(ABC):
    """Abstract base class for FederationManager."""

    @abstractmethod
    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""

    @abstractmethod
    def is_member(self, federation_name: str, flwr_aid: str) -> bool:
        """Check if a member of the federation."""

    @abstractmethod
    def filter_nodes(self, node_ids: set[int], federation_name: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""

    @abstractmethod
    def filter_messages(
        self, messages: list[Message], federation_name: str
    ) -> list[Message]:
        """Given a list of messages, filter out those from/to nodes outside the
        federation."""

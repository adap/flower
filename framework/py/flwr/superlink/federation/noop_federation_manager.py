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
"""NoOp implementation of FederationManager."""

from flwr.app import Message
from flwr.supercore.constant import NOOP_FEDERATION_NAME

from .federation_manager import FederationManager


class NoOpFederationManager(FederationManager):
    """No-Op FederationManager implementation."""

    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""
        return federation == NOOP_FEDERATION_NAME

    def is_member(self, federation: str, flwr_aid: str) -> bool:
        """Check if a member of the federation."""
        return True

    def filter_nodes(self, node_ids: set[int], federation: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""
        return node_ids

    def has_node(self, message: Message, federation: str) -> bool:
        """Given a message, check if it is from/to a node in the federation."""
        return True

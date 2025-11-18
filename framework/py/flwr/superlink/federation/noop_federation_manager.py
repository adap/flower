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

from flwr.common.constant import NOOP_FLWR_AID
from flwr.supercore.constant import NOOP_FEDERATION

from .federation_manager import FederationManager


class NoOpFederationManager(FederationManager):
    """No-Op FederationManager implementation."""

    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""
        return federation == NOOP_FEDERATION

    def has_member(self, flwr_aid: str, federation: str) -> bool:
        """Check if the given account is a member of the federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return flwr_aid == NOOP_FLWR_AID

    def filter_nodes(self, node_ids: set[int], federation: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return node_ids

    def has_node(self, node_id: int, federation: str) -> bool:
        """Given a node ID, check if it is in the federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return True

    def get_federations(self, flwr_aid: str) -> list[str]:
        """Get federations of which the account is a member."""
        if flwr_aid != NOOP_FLWR_AID:
            return []
        return [NOOP_FEDERATION]

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


from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.common.typing import Federation
from flwr.proto.federation_pb2 import (  # pylint: disable=E0611
    Account,
    Invitation,
    Member,
)
from flwr.supercore.constant import NOOP_FEDERATION, NOOP_FEDERATION_DESCRIPTION

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

    def get_federations(self, flwr_aid: str) -> list[Federation]:
        """Get federations of which the account is a member."""
        if flwr_aid != NOOP_FLWR_AID:
            return []
        return [
            Federation(
                name=NOOP_FEDERATION,
                description=NOOP_FEDERATION_DESCRIPTION,
                members=[],
                nodes=[],
                runs=[],
                archived=False,
            )
        ]

    def get_details(self, federation: str) -> Federation:
        """Get details of the federation."""
        if federation != NOOP_FEDERATION:
            raise ValueError(f"Federation '{federation}' does not exist.")

        runs = list(self.linkstate.get_run_info(flwr_aids=[NOOP_FLWR_AID]))
        nodes = list(self.linkstate.get_node_info(owner_aids=[NOOP_FLWR_AID]))
        only_account = Account(id=NOOP_FLWR_AID, name=NOOP_ACCOUNT_NAME)
        return Federation(
            name=NOOP_FEDERATION,
            description=NOOP_FEDERATION_DESCRIPTION,
            members=[
                Member(account=only_account, role="owner"),
            ],
            nodes=nodes,
            runs=runs,
            archived=False,
        )

    def create_federation(
        self, flwr_aid: str, name: str, description: str
    ) -> Federation:
        """Create a new federation."""
        raise NotImplementedError(
            "`create_federation` is not supported by NoOpFederationManager."
        )

    def archive_federation(self, flwr_aid: str, name: str) -> None:
        """Archive an existing federation."""
        raise NotImplementedError(
            "`archive_federation` is not supported by NoOpFederationManager."
        )

    def add_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Add a SuperNode to a federation."""
        raise NotImplementedError(
            "`add_supernode` is not supported by NoOpFederationManager."
        )

    def remove_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Remove a SuperNode from a federation."""
        raise NotImplementedError(
            "`remove_supernode` is not supported by NoOpFederationManager."
        )

    def create_invitation(
        self, flwr_aid: str, federation: str, invitee_flwr_aid: str
    ) -> None:
        """Create an invitation for an account to join a federation."""
        raise NotImplementedError(
            "`create_invitation` is not supported by NoOpFederationManager."
        )

    def list_invitations(
        self, flwr_aid: str
    ) -> tuple[list[Invitation], list[Invitation]]:
        """List invitations visible to the given account."""
        raise NotImplementedError(
            "`list_invitations` is not supported by NoOpFederationManager."
        )

    def accept_invitation(self, flwr_aid: str, federation: str) -> None:
        """Accept a pending invitation to join a federation."""
        raise NotImplementedError(
            "`accept_invitation` is not supported by NoOpFederationManager."
        )

    def reject_invitation(self, flwr_aid: str, federation: str) -> None:
        """Reject a pending invitation to join a federation."""
        raise NotImplementedError(
            "`reject_invitation` is not supported by NoOpFederationManager."
        )

    def revoke_invitation(
        self, flwr_aid: str, federation: str, invitee_flwr_aid: str
    ) -> None:
        """Revoke a pending invitation."""
        raise NotImplementedError(
            "`revoke_invitation` is not supported by NoOpFederationManager."
        )

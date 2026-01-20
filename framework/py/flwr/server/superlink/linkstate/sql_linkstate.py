# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""SQLAlchemy-based implementation of the link state."""


from collections.abc import Sequence

from sqlalchemy import MetaData

from flwr.app.user_config import UserConfig
from flwr.common import Context, Message
from flwr.common.record import ConfigRecord
from flwr.common.typing import Run, RunStatus
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.corestate.sql_corestate import SqlCoreState
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.superlink.federation import FederationManager

from .linkstate import LinkState


class SqlLinkState(LinkState, SqlCoreState):  # pylint: disable=R0904
    """SQLAlchemy-based LinkState implementation."""

    def __init__(
        self,
        database_path: str,
        federation_manager: FederationManager,
        object_store: ObjectStore,
    ) -> None:
        super().__init__(database_path, object_store)
        federation_manager.linkstate = self
        self._federation_manager = federation_manager

    def get_metadata(self) -> MetaData:
        """Return combined SQLAlchemy MetaData for LinkState and CoreState tables."""
        # Start with linkstate tables
        metadata = create_linkstate_metadata()

        # Add corestate tables (token_store)
        corestate_metadata = create_corestate_metadata()
        for table in corestate_metadata.tables.values():
            table.to_metadata(metadata)

        return metadata

    @property
    def federation_manager(self) -> FederationManager:
        """Return the FederationManager instance."""
        return self._federation_manager

    def store_message_ins(self, message: Message) -> str | None:
        """Store one Message."""
        raise NotImplementedError

    def get_message_ins(self, node_id: int, limit: int | None) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        raise NotImplementedError

    def store_message_res(self, message: Message) -> str | None:
        """Store one Message."""
        raise NotImplementedError

    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        raise NotImplementedError

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store."""
        raise NotImplementedError

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store."""
        raise NotImplementedError

    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        raise NotImplementedError

    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""
        raise NotImplementedError

    def create_node(
        self,
        owner_aid: str,
        owner_name: str,
        public_key: bytes,
        heartbeat_interval: float,
    ) -> int:
        """Create, store in the link state, and return `node_id`."""
        raise NotImplementedError

    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Delete a node."""
        raise NotImplementedError

    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`."""
        raise NotImplementedError

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        raise NotImplementedError

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set."""
        raise NotImplementedError

    def get_node_info(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        raise NotImplementedError

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        raise NotImplementedError

    def create_run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation: str,
        federation_options: ConfigRecord,
        flwr_aid: str | None,
    ) -> int:
        """Create a new run."""
        raise NotImplementedError

    def get_run_ids(self, flwr_aid: str | None) -> set[int]:
        """Retrieve all run IDs if `flwr_aid` is not specified.

        Otherwise, retrieve all run IDs for the specified `flwr_aid`.
        """
        raise NotImplementedError

    def get_run(self, run_id: int) -> Run | None:
        """Retrieve information about the run with the specified `run_id`."""
        raise NotImplementedError

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        raise NotImplementedError

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        raise NotImplementedError

    def get_pending_run_id(self) -> int | None:
        """Get the `run_id` of a run with `Status.PENDING` status."""
        raise NotImplementedError

    def get_federation_options(self, run_id: int) -> ConfigRecord | None:
        """Retrieve the federation options for the specified `run_id`."""
        raise NotImplementedError

    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node."""
        raise NotImplementedError

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Transition runs with expired tokens to failed status.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """

    def get_serverapp_context(self, run_id: int) -> Context | None:
        """Get the context for the specified `run_id`."""
        raise NotImplementedError

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        raise NotImplementedError

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`."""
        raise NotImplementedError

    def get_serverapp_log(
        self, run_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`."""
        raise NotImplementedError

    def store_traffic(self, run_id: int, *, bytes_sent: int, bytes_recv: int) -> None:
        """Store traffic data for the specified `run_id`."""
        raise NotImplementedError

    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`."""
        raise NotImplementedError

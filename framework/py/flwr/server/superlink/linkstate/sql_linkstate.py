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
from logging import ERROR
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.exc import IntegrityError

from flwr.app.user_config import UserConfig
from flwr.common import Context, Message, log, now
from flwr.common.constant import NODE_ID_NUM_BYTES, SUPERLINK_NODE_ID
from flwr.common.record import ConfigRecord
from flwr.common.typing import Run, RunStatus
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.constant import NodeStatus
from flwr.supercore.corestate.sql_corestate import SqlCoreState
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64
from flwr.superlink.federation import FederationManager

from .linkstate import LinkState
from .utils import generate_rand_int_from_bytes


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
        # Sample a random uint64 as node_id
        uint64_node_id = generate_rand_int_from_bytes(
            NODE_ID_NUM_BYTES, exclude=[SUPERLINK_NODE_ID, 0]
        )

        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(uint64_node_id)

        query = """
            INSERT INTO node
            (node_id, owner_aid, owner_name, status, registered_at, last_activated_at,
            last_deactivated_at, unregistered_at, online_until, heartbeat_interval,
            public_key)
            VALUES (:node_id, :owner_aid, :owner_name, :status, :registered_at,
            :last_activated_at, :last_deactivated_at, :unregistered_at, :online_until,
            :heartbeat_interval, :public_key)
        """

        # Mark the node online until now().timestamp() + heartbeat_interval
        try:
            self.query(
                query,
                {
                    "node_id": sint64_node_id,
                    "owner_aid": owner_aid,
                    "owner_name": owner_name,
                    "status": NodeStatus.REGISTERED,
                    "registered_at": now().isoformat(),
                    "last_activated_at": None,
                    "last_deactivated_at": None,
                    "unregistered_at": None,
                    "online_until": None,  # initialized with offline status
                    "heartbeat_interval": heartbeat_interval,
                    "public_key": public_key,
                },
            )
        except IntegrityError as e:
            if "node.public_key" in str(e):
                raise ValueError("Public key already in use.") from None
            # Must be node ID conflict, almost impossible unless system is compromised
            log(ERROR, "Unexpected node registration failure.")
            return 0

        # Note: we need to return the uint64 value of the node_id
        return uint64_node_id

    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Delete a node."""
        sint64_node_id = uint64_to_int64(node_id)

        query = """
            UPDATE node
            SET status = :status, unregistered_at = :unregistered_at,
            online_until = IIF(online_until > :current, :current2, online_until)
            WHERE node_id = :node_id AND status != :status2 AND owner_aid = :owner_aid
            RETURNING node_id
        """
        current = now()
        params = {
            "status": NodeStatus.UNREGISTERED,
            "unregistered_at": current.isoformat(),
            "current": current.timestamp(),
            "current2": current.timestamp(),
            "node_id": sint64_node_id,
            "status2": NodeStatus.UNREGISTERED,
            "owner_aid": owner_aid,
        }

        rows = self.query(query, params)
        if not rows:
            raise ValueError(
                f"Node {node_id} already deleted, not found or unauthorized "
                "deletion attempt."
            )

    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`."""
        raise NotImplementedError

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        raise NotImplementedError

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        with self.session():
            # Convert the uint64 value to sint64 for SQLite
            sint64_run_id = uint64_to_int64(run_id)

            # Validate run ID
            query = "SELECT federation FROM run WHERE run_id = :run_id"
            rows = self.query(query, {"run_id": sint64_run_id})
            if not rows:
                return set()
            federation: str = rows[0]["federation"]

            # Retrieve all online nodes
            node_ids = {
                node.node_id
                for node in self.get_node_info(statuses=[NodeStatus.ONLINE])
            }
        # Filter node IDs by federation
        return self.federation_manager.filter_nodes(node_ids, federation)

    def _check_and_tag_offline_nodes(self, node_ids: list[int] | None = None) -> None:
        """Check and tag offline nodes."""
        # strftime will convert POSIX timestamp to ISO format
        query = """
            UPDATE node SET status = :offline,
            last_deactivated_at =
            strftime('%Y-%m-%dT%H:%M:%f+00:00', online_until, 'unixepoch')
            WHERE online_until < :current_time AND status = :online
        """
        params: dict[str, Any] = {
            "offline": NodeStatus.OFFLINE,
            "current_time": now().timestamp(),
            "online": NodeStatus.ONLINE,
        }
        if node_ids is not None:
            placeholders = ",".join([f":nid_{i}" for i in range(len(node_ids))])
            query += f" AND node_id IN ({placeholders})"
            params.update(
                {f"nid_{i}": uint64_to_int64(nid) for i, nid in enumerate(node_ids)}
            )
        self.query(query, params)

    def get_node_info(  # pylint: disable=too-many-locals
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        with self.session():
            self._check_and_tag_offline_nodes()

            # Build the WHERE clause based on provided filters
            conditions = []
            params: dict[str, Any] = {}
            if node_ids is not None:
                sint64_node_ids = [uint64_to_int64(node_id) for node_id in node_ids]
                placeholders = ",".join(
                    [f":nid_{i}" for i in range(len(sint64_node_ids))]
                )
                conditions.append(f"node_id IN ({placeholders})")
                for i, nid in enumerate(sint64_node_ids):
                    params[f"nid_{i}"] = nid
            if owner_aids is not None:
                placeholders = ",".join([f":aid_{i}" for i in range(len(owner_aids))])
                conditions.append(f"owner_aid IN ({placeholders})")
                for i, aid in enumerate(owner_aids):
                    params[f"aid_{i}"] = aid
            if statuses is not None:
                placeholders = ",".join([f":st_{i}" for i in range(len(statuses))])
                conditions.append(f"status IN ({placeholders})")
                for i, status in enumerate(statuses):
                    params[f"st_{i}"] = status

            # Construct the final query
            query = "SELECT * FROM node"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            rows = self.query(query, params)

            result: list[NodeInfo] = []
            for row in rows:
                # Convert sint64 node_id to uint64
                row["node_id"] = int64_to_uint64(row["node_id"])
                result.append(NodeInfo(**row))

            return result

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        query = """SELECT node_id FROM node
                   WHERE public_key = :public_key AND status != :status;"""
        rows = self.query(
            query, {"public_key": public_key, "status": NodeStatus.UNREGISTERED}
        )

        # If no result is found, return None
        if not rows:
            return None

        # Convert sint64 node_id to uint64
        node_id = int64_to_uint64(rows[0]["node_id"])
        return node_id

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
        """Handle cleanup of expired tokens.

        Override in subclasses to add custom cleanup logic.

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

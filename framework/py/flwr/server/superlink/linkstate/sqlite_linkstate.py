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
"""SQLite based implemenation of the link state."""


# pylint: disable=too-many-lines

import json
import secrets
import sqlite3
from collections.abc import Sequence
from logging import ERROR, WARNING
from typing import Any, Optional, Union, cast

from flwr.common import Context, Message, Metadata, log, now
from flwr.common.constant import (
    FLWR_APP_TOKEN_LENGTH,
    HEARTBEAT_MAX_INTERVAL,
    HEARTBEAT_PATIENCE,
    MESSAGE_TTL_TOLERANCE,
    NODE_ID_NUM_BYTES,
    RUN_FAILURE_DETAILS_NO_HEARTBEAT,
    RUN_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    Status,
    SubStatus,
)
from flwr.common.message import make_message
from flwr.common.record import ConfigRecord
from flwr.common.serde import recorddict_from_proto, recorddict_to_proto
from flwr.common.serde_utils import error_from_proto, error_to_proto
from flwr.common.typing import Run, RunStatus, UserConfig

# pylint: disable=E0611
from flwr.proto.error_pb2 import Error as ProtoError
from flwr.proto.node_pb2 import NodeInfo
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict

# pylint: enable=E0611
from flwr.server.utils.validator import validate_message
from flwr.supercore.constant import NodeStatus
from flwr.supercore.sqlite_mixin import SqliteMixin
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64

from .linkstate import LinkState
from .utils import (
    check_node_availability_for_in_message,
    configrecord_from_bytes,
    configrecord_to_bytes,
    context_from_bytes,
    context_to_bytes,
    convert_sint64_values_in_dict_to_uint64,
    convert_uint64_values_in_dict_to_sint64,
    generate_rand_int_from_bytes,
    has_valid_sub_status,
    is_valid_transition,
    verify_found_message_replies,
    verify_message_ids,
)

SQL_CREATE_TABLE_NODE = """
CREATE TABLE IF NOT EXISTS node(
    node_id                 INTEGER UNIQUE,
    owner_aid               TEXT,
    status                  TEXT,
    registered_at           TEXT,
    last_activated_at       TEXT NULL,
    last_deactivated_at     TEXT NULL,
    unregistered_at         TEXT NULL,
    online_until            TIMESTAMP NULL,
    heartbeat_interval      REAL,
    public_key              BLOB UNIQUE
);
"""

SQL_CREATE_TABLE_PUBLIC_KEY = """
CREATE TABLE IF NOT EXISTS public_key(
    public_key      BLOB PRIMARY KEY
);
"""

SQL_CREATE_INDEX_ONLINE_UNTIL = """
CREATE INDEX IF NOT EXISTS idx_online_until ON node (online_until);
"""

SQL_CREATE_INDEX_OWNER_AID = """
CREATE INDEX IF NOT EXISTS idx_node_owner_aid ON node(owner_aid);
"""

SQL_CREATE_TABLE_RUN = """
CREATE TABLE IF NOT EXISTS run(
    run_id                INTEGER UNIQUE,
    active_until          REAL,
    heartbeat_interval    REAL,
    fab_id                TEXT,
    fab_version           TEXT,
    fab_hash              TEXT,
    override_config       TEXT,
    pending_at            TEXT,
    starting_at           TEXT,
    running_at            TEXT,
    finished_at           TEXT,
    sub_status            TEXT,
    details               TEXT,
    federation_options    BLOB,
    flwr_aid              TEXT
);
"""

SQL_CREATE_TABLE_LOGS = """
CREATE TABLE IF NOT EXISTS logs (
    timestamp             REAL,
    run_id                INTEGER,
    node_id               INTEGER,
    log                   TEXT,
    PRIMARY KEY (timestamp, run_id, node_id),
    FOREIGN KEY (run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_CONTEXT = """
CREATE TABLE IF NOT EXISTS context(
    run_id                INTEGER UNIQUE,
    context               BLOB,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_MESSAGE_INS = """
CREATE TABLE IF NOT EXISTS message_ins(
    message_id              TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    src_node_id             INTEGER,
    dst_node_id             INTEGER,
    reply_to_message_id     TEXT,
    created_at              REAL,
    delivered_at            TEXT,
    ttl                     REAL,
    message_type            TEXT,
    content                 BLOB NULL,
    error                   BLOB NULL,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""


SQL_CREATE_TABLE_MESSAGE_RES = """
CREATE TABLE IF NOT EXISTS message_res(
    message_id              TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    src_node_id             INTEGER,
    dst_node_id             INTEGER,
    reply_to_message_id     TEXT,
    created_at              REAL,
    delivered_at            TEXT,
    ttl                     REAL,
    message_type            TEXT,
    content                 BLOB NULL,
    error                   BLOB NULL,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_TOKEN_STORE = """
CREATE TABLE IF NOT EXISTS token_store (
    run_id                  INTEGER PRIMARY KEY,
    token                   TEXT UNIQUE NOT NULL
);
"""


class SqliteLinkState(LinkState, SqliteMixin):  # pylint: disable=R0904
    """SQLite-based LinkState implementation."""

    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Connect to the DB, enable FK support, and create tables if needed."""
        return self._ensure_initialized(
            SQL_CREATE_TABLE_RUN,
            SQL_CREATE_TABLE_LOGS,
            SQL_CREATE_TABLE_CONTEXT,
            SQL_CREATE_TABLE_MESSAGE_INS,
            SQL_CREATE_TABLE_MESSAGE_RES,
            SQL_CREATE_TABLE_NODE,
            SQL_CREATE_TABLE_PUBLIC_KEY,
            SQL_CREATE_TABLE_TOKEN_STORE,
            SQL_CREATE_INDEX_ONLINE_UNTIL,
            SQL_CREATE_INDEX_OWNER_AID,
            log_queries=log_queries,
        )

    def store_message_ins(self, message: Message) -> Optional[str]:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None

        # Store Message
        data = (message_to_dict(message),)

        # Convert values from uint64 to sint64 for SQLite
        convert_uint64_values_in_dict_to_sint64(
            data[0], ["run_id", "src_node_id", "dst_node_id"]
        )

        # Validate run_id
        query = "SELECT run_id FROM run WHERE run_id = ?;"
        if not self.query(query, (data[0]["run_id"],)):
            log(ERROR, "Invalid run ID for Message: %s", message.metadata.run_id)
            return None

        # Validate source node ID
        if message.metadata.src_node_id != SUPERLINK_NODE_ID:
            log(
                ERROR,
                "Invalid source node ID for Message: %s",
                message.metadata.src_node_id,
            )
            return None

        # Validate destination node ID
        query = "SELECT node_id FROM node WHERE node_id = ?;"
        if not self.query(query, (data[0]["dst_node_id"],)):
            log(
                ERROR,
                "Invalid destination node ID for Message: %s",
                message.metadata.dst_node_id,
            )
            return None

        columns = ", ".join([f":{key}" for key in data[0]])
        query = f"INSERT INTO message_ins VALUES({columns});"

        # Only invalid run_id can trigger IntegrityError.
        # This may need to be changed in the future version with more integrity checks.
        self.query(query, data)

        return message.metadata.message_id

    def get_message_ins(self, node_id: int, limit: Optional[int]) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        if node_id == SUPERLINK_NODE_ID:
            msg = f"`node_id` must be != {SUPERLINK_NODE_ID}"
            raise AssertionError(msg)

        data: dict[str, Union[str, int]] = {}

        # Convert the uint64 value to sint64 for SQLite
        data["node_id"] = uint64_to_int64(node_id)

        # Retrieve all Messages for node_id
        query = """
            SELECT message_id
            FROM message_ins
            WHERE   dst_node_id == :node_id
            AND   delivered_at = ""
            AND   (created_at + ttl) > CAST(strftime('%s', 'now') AS REAL)
        """

        if limit is not None:
            query += " LIMIT :limit"
            data["limit"] = limit

        query += ";"

        rows = self.query(query, data)

        if rows:
            # Prepare query
            message_ids = [row["message_id"] for row in rows]
            placeholders: str = ",".join([f":id_{i}" for i in range(len(message_ids))])
            query = f"""
                UPDATE message_ins
                SET delivered_at = :delivered_at
                WHERE message_id IN ({placeholders})
                RETURNING *;
            """

            # Prepare data for query
            delivered_at = now().isoformat()
            data = {"delivered_at": delivered_at}
            for index, msg_id in enumerate(message_ids):
                data[f"id_{index}"] = str(msg_id)

            # Run query
            rows = self.query(query, data)

        for row in rows:
            # Convert values from sint64 to uint64
            convert_sint64_values_in_dict_to_uint64(
                row, ["run_id", "src_node_id", "dst_node_id"]
            )

        result = [dict_to_message(row) for row in rows]

        return result

    def store_message_res(self, message: Message) -> Optional[str]:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=True)
        if any(errors):
            log(ERROR, errors)
            return None

        res_metadata = message.metadata
        msg_ins_id = res_metadata.reply_to_message_id
        msg_ins = self.get_valid_message_ins(msg_ins_id)
        if msg_ins is None:
            log(
                ERROR,
                "Failed to store Message reply: "
                "The message it replies to with message_id %s does not exist or "
                "has expired.",
                msg_ins_id,
            )
            return None

        # Ensure that the dst_node_id of the original message matches the src_node_id of
        # reply being processed.
        if (
            msg_ins
            and message
            and int64_to_uint64(msg_ins["dst_node_id"]) != res_metadata.src_node_id
        ):
            return None

        # Fail if the Message TTL exceeds the
        # expiration time of the Message it replies to.
        # Condition: ins_metadata.created_at + ins_metadata.ttl ≥
        #            res_metadata.created_at + res_metadata.ttl
        # A small tolerance is introduced to account
        # for floating-point precision issues.
        max_allowed_ttl = (
            msg_ins["created_at"] + msg_ins["ttl"] - res_metadata.created_at
        )
        if res_metadata.ttl and (
            res_metadata.ttl - max_allowed_ttl > MESSAGE_TTL_TOLERANCE
        ):
            log(
                WARNING,
                "Received Message with TTL %.2f exceeding the allowed maximum "
                "TTL %.2f.",
                res_metadata.ttl,
                max_allowed_ttl,
            )
            return None

        # Store Message
        data = (message_to_dict(message),)

        # Convert values from uint64 to sint64 for SQLite
        convert_uint64_values_in_dict_to_sint64(
            data[0], ["run_id", "src_node_id", "dst_node_id"]
        )

        columns = ", ".join([f":{key}" for key in data[0]])
        query = f"INSERT INTO message_res VALUES({columns});"

        # Only invalid run_id can trigger IntegrityError.
        # This may need to be changed in the future version with more integrity checks.
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            log(ERROR, "`run` is invalid")
            return None

        return message.metadata.message_id

    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        # pylint: disable-msg=too-many-locals
        ret: dict[str, Message] = {}

        # Verify Message IDs
        current = now().timestamp()
        query = f"""
            SELECT *
            FROM message_ins
            WHERE message_id IN ({",".join(["?"] * len(message_ids))});
        """
        rows = self.query(query, tuple(str(message_id) for message_id in message_ids))
        found_message_ins_dict: dict[str, Message] = {}
        for row in rows:
            convert_sint64_values_in_dict_to_uint64(
                row, ["run_id", "src_node_id", "dst_node_id"]
            )
            found_message_ins_dict[row["message_id"]] = dict_to_message(row)

        ret = verify_message_ids(
            inquired_message_ids=message_ids,
            found_message_ins_dict=found_message_ins_dict,
            current_time=current,
        )

        # Check node availability
        dst_node_ids: set[int] = set()
        for message_id in message_ids:
            in_message = found_message_ins_dict[message_id]
            sint_node_id = uint64_to_int64(in_message.metadata.dst_node_id)
            dst_node_ids.add(sint_node_id)
        query = f"""
            SELECT node_id, online_until
            FROM node
            WHERE node_id IN ({",".join(["?"] * len(dst_node_ids))})
            AND status != ?
        """
        rows = self.query(query, tuple(dst_node_ids) + (NodeStatus.UNREGISTERED,))
        tmp_ret_dict = check_node_availability_for_in_message(
            inquired_in_message_ids=message_ids,
            found_in_message_dict=found_message_ins_dict,
            node_id_to_online_until={
                int64_to_uint64(row["node_id"]): row["online_until"] for row in rows
            },
            current_time=current,
        )
        ret.update(tmp_ret_dict)

        # Find all reply Messages
        query = f"""
            SELECT *
            FROM message_res
            WHERE reply_to_message_id IN ({",".join(["?"] * len(message_ids))})
            AND delivered_at = "";
        """
        rows = self.query(query, tuple(str(message_id) for message_id in message_ids))
        for row in rows:
            convert_sint64_values_in_dict_to_uint64(
                row, ["run_id", "src_node_id", "dst_node_id"]
            )
        tmp_ret_dict = verify_found_message_replies(
            inquired_message_ids=message_ids,
            found_message_ins_dict=found_message_ins_dict,
            found_message_res_list=[dict_to_message(row) for row in rows],
            current_time=current,
        )
        ret.update(tmp_ret_dict)

        # Mark existing reply Messages to be returned as delivered
        delivered_at = now().isoformat()
        for message_res in ret.values():
            message_res.metadata.delivered_at = delivered_at
        message_res_ids = [
            message_res.metadata.message_id for message_res in ret.values()
        ]
        query = f"""
            UPDATE message_res
            SET delivered_at = ?
            WHERE message_id IN ({",".join(["?"] * len(message_res_ids))});
        """
        data: list[Any] = [delivered_at] + message_res_ids
        self.query(query, data)

        return list(ret.values())

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_ins;"
        rows = self.query(query)
        result = rows[0]
        num = cast(int, result["num"])
        return num

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_res;"
        rows = self.query(query)
        result: dict[str, int] = rows[0]
        return result["num"]

    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        if not message_ins_ids:
            return
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        placeholders = ",".join(["?"] * len(message_ins_ids))
        data = tuple(str(message_id) for message_id in message_ins_ids)

        # Delete Message
        query_1 = f"""
            DELETE FROM message_ins
            WHERE message_id IN ({placeholders});
        """

        # Delete reply Message
        query_2 = f"""
            DELETE FROM message_res
            WHERE reply_to_message_id IN ({placeholders});
        """

        with self.conn:
            self.conn.execute(query_1, data)
            self.conn.execute(query_2, data)

    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        query = """
            SELECT message_id
            FROM message_ins
            WHERE run_id = :run_id;
        """

        sint64_run_id = uint64_to_int64(run_id)
        data = {"run_id": sint64_run_id}

        with self.conn:
            rows = self.conn.execute(query, data).fetchall()

        return {row["message_id"] for row in rows}

    def create_node(
        self, owner_aid: str, public_key: bytes, heartbeat_interval: float
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
            (node_id, owner_aid, status, registered_at, last_activated_at,
            last_deactivated_at, unregistered_at, online_until, heartbeat_interval,
            public_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Mark the node online until now().timestamp() + heartbeat_interval
        try:
            self.query(
                query,
                (
                    sint64_node_id,  # node_id
                    owner_aid,  # owner_aid
                    NodeStatus.REGISTERED,  # status
                    now().isoformat(),  # registered_at
                    None,  # last_activated_at
                    None,  # last_deactivated_at
                    None,  # unregistered_at
                    None,  # online_until, initialized with offline status
                    heartbeat_interval,  # heartbeat_interval
                    public_key,  # public_key
                ),
            )
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: node.public_key" in str(e):
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
            SET status = ?, unregistered_at = ?,
            online_until = IIF(online_until > ?, ?, online_until)
            WHERE node_id = ? AND status != ? AND owner_aid = ?
            RETURNING node_id
        """
        current = now()
        params = (
            NodeStatus.UNREGISTERED,
            current.isoformat(),
            current.timestamp(),
            current.timestamp(),
            sint64_node_id,
            NodeStatus.UNREGISTERED,
            owner_aid,
        )

        rows = self.query(query, params)
        if not rows:
            raise ValueError(
                f"Node {node_id} already deleted, not found or unauthorized "
                "deletion attempt."
            )

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        # Validate run ID
        query = "SELECT COUNT(*) FROM run WHERE run_id = ?"
        rows = self.query(query, (sint64_run_id,))
        if rows[0]["COUNT(*)"] == 0:
            return set()

        # Retrieve all online nodes
        return {
            node.node_id for node in self.get_node_info(statuses=[NodeStatus.ONLINE])
        }

    def get_node_info(
        self,
        *,
        node_ids: Optional[Sequence[int]] = None,
        owner_aids: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        if self.conn is None:
            raise AttributeError("LinkState is not initialized.")

        with self.conn:
            # Check and tag offline nodes
            current_dt = now()
            # strftime will convert POSIX timestamp to ISO format
            query = """
                UPDATE node SET status = ?,
                last_deactivated_at =
                strftime("%Y-%m-%dT%H:%M:%f+00:00", online_until, "unixepoch")
                WHERE online_until <= ? AND status == ?
            """
            params: list[Any] = [
                NodeStatus.OFFLINE,
                current_dt.timestamp(),
                NodeStatus.ONLINE,
            ]
            self.conn.execute(query, params)

            # Build the WHERE clause based on provided filters
            conditions = []
            params = []
            if node_ids is not None:
                sint64_node_ids = [uint64_to_int64(node_id) for node_id in node_ids]
                placeholders = ",".join(["?"] * len(sint64_node_ids))
                conditions.append(f"node_id IN ({placeholders})")
                params.extend(sint64_node_ids)
            if owner_aids is not None:
                placeholders = ",".join(["?"] * len(owner_aids))
                conditions.append(f"owner_aid IN ({placeholders})")
                params.extend(owner_aids)
            if statuses is not None:
                placeholders = ",".join(["?"] * len(statuses))
                conditions.append(f"status IN ({placeholders})")
                params.extend(statuses)

            # Construct the final query
            query = "SELECT * FROM node"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            rows = self.conn.execute(query, params).fetchall()

            result: list[NodeInfo] = []
            for row in rows:
                # Convert sint64 node_id to uint64
                row["node_id"] = int64_to_uint64(row["node_id"])
                result.append(NodeInfo(**row))

            return result

    def get_node_public_key(self, node_id: int) -> bytes:
        """Get `public_key` for the specified `node_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(node_id)

        # Query the public key for the given node_id
        query = "SELECT public_key FROM node WHERE node_id = ? AND status != ?;"
        rows = self.query(query, (sint64_node_id, NodeStatus.UNREGISTERED))

        # If no result is found, return None
        if not rows:
            raise ValueError(f"Node ID {node_id} not found")

        # Return the public key
        return cast(bytes, rows[0]["public_key"])

    def get_node_id_by_public_key(self, public_key: bytes) -> Optional[int]:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        query = "SELECT node_id FROM node WHERE public_key = ? AND status != ?;"
        rows = self.query(query, (public_key, NodeStatus.UNREGISTERED))

        # If no result is found, return None
        if not rows:
            return None

        # Convert sint64 node_id to uint64
        node_id = int64_to_uint64(rows[0]["node_id"])
        return node_id

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_run(
        self,
        fab_id: Optional[str],
        fab_version: Optional[str],
        fab_hash: Optional[str],
        override_config: UserConfig,
        federation_options: ConfigRecord,
        flwr_aid: Optional[str],
    ) -> int:
        """Create a new run for the specified `fab_id` and `fab_version`."""
        # Sample a random int64 as run_id
        uint64_run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(uint64_run_id)

        # Check conflicts
        query = "SELECT COUNT(*) FROM run WHERE run_id = ?;"
        # If sint64_run_id does not exist
        if self.query(query, (sint64_run_id,))[0]["COUNT(*)"] == 0:
            query = (
                "INSERT INTO run "
                "(run_id, active_until, heartbeat_interval, fab_id, fab_version, "
                "fab_hash, override_config, federation_options, pending_at, "
                "starting_at, running_at, finished_at, sub_status, details, flwr_aid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
            )
            override_config_json = json.dumps(override_config)
            data = [
                sint64_run_id,
                0,  # The `active_until` is not used until the run is started
                0,  # This `heartbeat_interval` is not used until the run is started
                fab_id,
                fab_version,
                fab_hash,
                override_config_json,
                configrecord_to_bytes(federation_options),
                now().isoformat(),
                "",
                "",
                "",
                "",
                "",
                flwr_aid or "",
            ]
            self.query(query, tuple(data))
            return uint64_run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run_ids(self, flwr_aid: Optional[str]) -> set[int]:
        """Retrieve all run IDs if `flwr_aid` is not specified.

        Otherwise, retrieve all run IDs for the specified `flwr_aid`.
        """
        if flwr_aid:
            rows = self.query(
                "SELECT run_id FROM run WHERE flwr_aid = ?;",
                (flwr_aid,),
            )
        else:
            rows = self.query("SELECT run_id FROM run;", ())
        return {int64_to_uint64(row["run_id"]) for row in rows}

    def _check_and_tag_inactive_run(self, run_ids: set[int]) -> None:
        """Check if any runs are no longer active.

        Marks runs with status 'starting' or 'running' as failed
        if they have not sent a heartbeat before `active_until`.
        """
        sint_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
        query = "UPDATE run SET finished_at = ?, sub_status = ?, details = ? "
        query += "WHERE starting_at != '' AND finished_at = '' AND active_until < ?"
        query += f" AND run_id IN ({','.join(['?'] * len(run_ids))});"
        current = now()
        self.query(
            query,
            (
                current.isoformat(),
                SubStatus.FAILED,
                RUN_FAILURE_DETAILS_NO_HEARTBEAT,
                current.timestamp(),
                *sint_run_ids,
            ),
        )

    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve information about the run with the specified `run_id`."""
        # Check if runs are still active
        self._check_and_tag_inactive_run(run_ids={run_id})

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)
        query = "SELECT * FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint64_run_id,))
        if rows:
            row = rows[0]
            return Run(
                run_id=int64_to_uint64(row["run_id"]),
                fab_id=row["fab_id"],
                fab_version=row["fab_version"],
                fab_hash=row["fab_hash"],
                override_config=json.loads(row["override_config"]),
                pending_at=row["pending_at"],
                starting_at=row["starting_at"],
                running_at=row["running_at"],
                finished_at=row["finished_at"],
                status=RunStatus(
                    status=determine_run_status(row),
                    sub_status=row["sub_status"],
                    details=row["details"],
                ),
                flwr_aid=row["flwr_aid"],
            )
        log(ERROR, "`run_id` does not exist.")
        return None

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        # Check if runs are still active
        self._check_and_tag_inactive_run(run_ids=run_ids)

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_ids = (uint64_to_int64(run_id) for run_id in set(run_ids))
        query = f"SELECT * FROM run WHERE run_id IN ({','.join(['?'] * len(run_ids))});"
        rows = self.query(query, tuple(sint64_run_ids))

        return {
            # Restore uint64 run IDs
            int64_to_uint64(row["run_id"]): RunStatus(
                status=determine_run_status(row),
                sub_status=row["sub_status"],
                details=row["details"],
            )
            for row in rows
        }

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        # Check if runs are still active
        self._check_and_tag_inactive_run(run_ids={run_id})

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)
        query = "SELECT * FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint64_run_id,))

        # Check if the run_id exists
        if not rows:
            log(ERROR, "`run_id` is invalid")
            return False

        # Check if the status transition is valid
        row = rows[0]
        current_status = RunStatus(
            status=determine_run_status(row),
            sub_status=row["sub_status"],
            details=row["details"],
        )
        if not is_valid_transition(current_status, new_status):
            log(
                ERROR,
                'Invalid status transition: from "%s" to "%s"',
                current_status.status,
                new_status.status,
            )
            return False

        # Check if the sub-status is valid
        if not has_valid_sub_status(current_status):
            log(
                ERROR,
                'Invalid sub-status "%s" for status "%s"',
                current_status.sub_status,
                current_status.status,
            )
            return False

        # Update the status
        query = "UPDATE run SET %s= ?, sub_status = ?, details = ?, "
        query += "active_until = ?, heartbeat_interval = ? "
        query += "WHERE run_id = ?;"

        # Prepare data for query
        # Initialize heartbeat_interval and active_until
        # when switching to starting or running
        current = now()
        if new_status.status in (Status.STARTING, Status.RUNNING):
            heartbeat_interval = HEARTBEAT_MAX_INTERVAL
            active_until = current.timestamp() + heartbeat_interval
        else:
            heartbeat_interval = 0
            active_until = 0

        # Determine the timestamp field based on the new status
        timestamp_fld = ""
        if new_status.status == Status.STARTING:
            timestamp_fld = "starting_at"
        elif new_status.status == Status.RUNNING:
            timestamp_fld = "running_at"
        elif new_status.status == Status.FINISHED:
            timestamp_fld = "finished_at"

        data = (
            current.isoformat(),
            new_status.sub_status,
            new_status.details,
            active_until,
            heartbeat_interval,
            uint64_to_int64(run_id),
        )
        self.query(query % timestamp_fld, data)
        return True

    def get_pending_run_id(self) -> Optional[int]:
        """Get the `run_id` of a run with `Status.PENDING` status, if any."""
        pending_run_id = None

        # Fetch all runs with unset `starting_at` (i.e. they are in PENDING status)
        query = "SELECT * FROM run WHERE starting_at = '' LIMIT 1;"
        rows = self.query(query)
        if rows:
            pending_run_id = int64_to_uint64(rows[0]["run_id"])

        return pending_run_id

    def get_federation_options(self, run_id: int) -> Optional[ConfigRecord]:
        """Retrieve the federation options for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)
        query = "SELECT federation_options FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint64_run_id,))

        # Check if the run_id exists
        if not rows:
            log(ERROR, "`run_id` is invalid")
            return None

        row = rows[0]
        return configrecord_from_bytes(row["federation_options"])

    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node, serving as a heartbeat.

        A node is considered online as long as it sends heartbeats within
        the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before
        the node is marked as offline.
        """
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        sint64_node_id = uint64_to_int64(node_id)

        with self.conn:
            # Check if node exists and not deleted
            query = "SELECT status FROM node WHERE node_id = ? AND status != ?"
            row = self.conn.execute(
                query, (sint64_node_id, NodeStatus.UNREGISTERED)
            ).fetchone()
            if row is None:
                return False

            # Construct query and params
            current_dt = now()
            query = "UPDATE node SET online_until = ?, heartbeat_interval = ?"
            params: list[Any] = [
                current_dt.timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval,
                heartbeat_interval,
            ]

            # Set timestamp if the status changes
            if row["status"] != NodeStatus.ONLINE:
                query += ", status = ?, last_activated_at = ?"
                params += [NodeStatus.ONLINE, current_dt.isoformat()]

            # Execute the query, refreshing `online_until` and `heartbeat_interval`
            query += " WHERE node_id = ?"
            params += [sint64_node_id]
            self.conn.execute(query, params)
            return True

    def acknowledge_app_heartbeat(self, run_id: int, heartbeat_interval: float) -> bool:
        """Acknowledge a heartbeat received from a ServerApp for a given run.

        A run with status `"running"` is considered alive as long as it sends heartbeats
        within the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before the run is
        marked as `"completed:failed"`.
        """
        # Check if runs are still active
        self._check_and_tag_inactive_run(run_ids={run_id})

        # Search for the run
        sint_run_id = uint64_to_int64(run_id)
        query = "SELECT * FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint_run_id,))

        if not rows:
            log(ERROR, "`run_id` is invalid")
            return False

        # Check if the run is of status "running"/"starting"
        row = rows[0]
        status = determine_run_status(row)
        if status not in (Status.RUNNING, Status.STARTING):
            log(
                ERROR,
                'Cannot acknowledge heartbeat for run with status "%s"',
                status,
            )
            return False

        # Update the `active_until` and `heartbeat_interval` for the given run
        active_until = now().timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval
        query = "UPDATE run SET active_until = ?, heartbeat_interval = ? "
        query += "WHERE run_id = ?"
        self.query(query, (active_until, heartbeat_interval, sint_run_id))
        return True

    def get_serverapp_context(self, run_id: int) -> Optional[Context]:
        """Get the context for the specified `run_id`."""
        # Retrieve context if any
        query = "SELECT context FROM context WHERE run_id = ?;"
        rows = self.query(query, (uint64_to_int64(run_id),))
        context = context_from_bytes(rows[0]["context"]) if rows else None
        return context

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        # Convert context to bytes
        context_bytes = context_to_bytes(context)
        sint_run_id = uint64_to_int64(run_id)

        # Check if any existing Context assigned to the run_id
        query = "SELECT COUNT(*) FROM context WHERE run_id = ?;"
        if self.query(query, (sint_run_id,))[0]["COUNT(*)"] > 0:
            # Update context
            query = "UPDATE context SET context = ? WHERE run_id = ?;"
            self.query(query, (context_bytes, sint_run_id))
        else:
            try:
                # Store context
                query = "INSERT INTO context (run_id, context) VALUES (?, ?);"
                self.query(query, (sint_run_id, context_bytes))
            except sqlite3.IntegrityError:
                raise ValueError(f"Run {run_id} not found") from None

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        # Store log
        try:
            query = """
                INSERT INTO logs (timestamp, run_id, node_id, log) VALUES (?, ?, ?, ?);
            """
            self.query(query, (now().timestamp(), sint64_run_id, 0, log_message))
        except sqlite3.IntegrityError:
            raise ValueError(f"Run {run_id} not found") from None

    def get_serverapp_log(
        self, run_id: int, after_timestamp: Optional[float]
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        # Check if the run_id exists
        query = "SELECT run_id FROM run WHERE run_id = ?;"
        if not self.query(query, (sint64_run_id,)):
            raise ValueError(f"Run {run_id} not found")

        # Retrieve logs
        if after_timestamp is None:
            after_timestamp = 0.0
        query = """
            SELECT log, timestamp FROM logs
            WHERE run_id = ? AND node_id = ? AND timestamp > ?;
        """
        rows = self.query(query, (sint64_run_id, 0, after_timestamp))
        rows.sort(key=lambda x: x["timestamp"])
        latest_timestamp = rows[-1]["timestamp"] if rows else 0.0
        return "".join(row["log"] for row in rows), latest_timestamp

    def get_valid_message_ins(self, message_id: str) -> Optional[dict[str, Any]]:
        """Check if the Message exists and is valid (not expired).

        Return Message if valid.
        """
        query = """
            SELECT *
            FROM message_ins
            WHERE message_id = :message_id
        """
        data = {"message_id": message_id}
        rows = self.query(query, data)
        if not rows:
            # Message does not exist
            return None

        message_ins = rows[0]
        created_at = message_ins["created_at"]
        ttl = message_ins["ttl"]
        current_time = now().timestamp()

        # Check if Message is expired
        if ttl is not None and created_at + ttl <= current_time:
            return None

        return message_ins

    def create_token(self, run_id: int) -> Optional[str]:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        query = "INSERT INTO token_store (run_id, token) VALUES (:run_id, :token);"
        data = {"run_id": uint64_to_int64(run_id), "token": token}
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            return None  # Token already created for this run ID
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        query = "SELECT token FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        rows = self.query(query, data)
        if not rows:
            return False
        return cast(str, rows[0]["token"]) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        query = "DELETE FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        self.query(query, data)

    def get_run_id_by_token(self, token: str) -> Optional[int]:
        """Get the run ID associated with a given token."""
        query = "SELECT run_id FROM token_store WHERE token = :token;"
        data = {"token": token}
        rows = self.query(query, data)
        if not rows:
            return None
        return int64_to_uint64(rows[0]["run_id"])


def message_to_dict(message: Message) -> dict[str, Any]:
    """Transform Message to dict."""
    result = {
        "message_id": message.metadata.message_id,
        "group_id": message.metadata.group_id,
        "run_id": message.metadata.run_id,
        "src_node_id": message.metadata.src_node_id,
        "dst_node_id": message.metadata.dst_node_id,
        "reply_to_message_id": message.metadata.reply_to_message_id,
        "created_at": message.metadata.created_at,
        "delivered_at": message.metadata.delivered_at,
        "ttl": message.metadata.ttl,
        "message_type": message.metadata.message_type,
        "content": None,
        "error": None,
    }

    if message.has_content():
        result["content"] = recorddict_to_proto(message.content).SerializeToString()
    else:
        result["error"] = error_to_proto(message.error).SerializeToString()

    return result


def dict_to_message(message_dict: dict[str, Any]) -> Message:
    """Transform dict to Message."""
    content, error = None, None
    if (b_content := message_dict.pop("content")) is not None:
        content = recorddict_from_proto(ProtoRecordDict.FromString(b_content))
    if (b_error := message_dict.pop("error")) is not None:
        error = error_from_proto(ProtoError.FromString(b_error))

    # Metadata constructor doesn't allow passing created_at. We set it later
    metadata = Metadata(
        **{k: v for k, v in message_dict.items() if k not in ["delivered_at"]}
    )
    msg = make_message(metadata=metadata, content=content, error=error)
    msg.metadata.delivered_at = message_dict["delivered_at"]
    return msg


def determine_run_status(row: dict[str, Any]) -> str:
    """Determine the status of the run based on timestamp fields."""
    if row["pending_at"]:
        if row["finished_at"]:
            return Status.FINISHED
        if row["starting_at"]:
            if row["running_at"]:
                return Status.RUNNING
            return Status.STARTING
        return Status.PENDING
    run_id = int64_to_uint64(row["run_id"])
    raise sqlite3.IntegrityError(f"The run {run_id} does not have a valid status.")

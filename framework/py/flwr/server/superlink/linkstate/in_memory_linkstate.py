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
"""In-memory LinkState implementation."""


import threading
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from logging import ERROR, WARNING
from typing import Optional
from uuid import UUID, uuid4

from flwr.common import Context, Message, log, now
from flwr.common.constant import (
    MESSAGE_TTL_TOLERANCE,
    NODE_ID_NUM_BYTES,
    PING_PATIENCE,
    RUN_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.record import ConfigRecord
from flwr.common.typing import Run, RunStatus, UserConfig
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.utils import validate_message

from .utils import (
    check_node_availability_for_in_message,
    generate_rand_int_from_bytes,
    has_valid_sub_status,
    is_valid_transition,
    verify_found_message_replies,
    verify_message_ids,
)


@dataclass
class RunRecord:  # pylint: disable=R0902
    """The record of a specific run, including its status and timestamps."""

    run: Run
    logs: list[tuple[float, str]] = field(default_factory=list)
    log_lock: threading.Lock = field(default_factory=threading.Lock)


class InMemoryLinkState(LinkState):  # pylint: disable=R0902,R0904
    """In-memory LinkState implementation."""

    def __init__(self) -> None:

        # Map node_id to (online_until, ping_interval)
        self.node_ids: dict[int, tuple[float, float]] = {}
        self.public_key_to_node_id: dict[bytes, int] = {}
        self.node_id_to_public_key: dict[int, bytes] = {}

        # Map run_id to RunRecord
        self.run_ids: dict[int, RunRecord] = {}
        self.contexts: dict[int, Context] = {}
        self.federation_options: dict[int, ConfigRecord] = {}
        self.message_ins_store: dict[UUID, Message] = {}
        self.message_res_store: dict[UUID, Message] = {}
        self.message_ins_id_to_message_res_id: dict[UUID, UUID] = {}

        self.node_public_keys: set[bytes] = set()

        self.lock = threading.RLock()

    def store_message_ins(self, message: Message) -> Optional[UUID]:
        """Store one Message."""
        # Validate message
        errors = validate_message(message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None
        # Validate run_id
        if message.metadata.run_id not in self.run_ids:
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
        if message.metadata.dst_node_id not in self.node_ids:
            log(
                ERROR,
                "Invalid destination node ID for Message: %s",
                message.metadata.dst_node_id,
            )
            return None

        # Create message_id
        message_id = uuid4()

        # Store Message
        # pylint: disable-next=W0212
        message.metadata._message_id = str(message_id)  # type: ignore
        with self.lock:
            self.message_ins_store[message_id] = message

        # Return the new message_id
        return message_id

    def get_message_ins(self, node_id: int, limit: Optional[int]) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find Message for node_id that were not delivered yet
        message_ins_list: list[Message] = []
        current_time = time.time()
        with self.lock:
            for _, msg_ins in self.message_ins_store.items():
                if (
                    msg_ins.metadata.dst_node_id == node_id
                    and msg_ins.metadata.delivered_at == ""
                    and msg_ins.metadata.created_at + msg_ins.metadata.ttl
                    > current_time
                ):
                    message_ins_list.append(msg_ins)
                if limit and len(message_ins_list) == limit:
                    break

        # Mark all of them as delivered
        delivered_at = now().isoformat()
        for msg_ins in message_ins_list:
            msg_ins.metadata.delivered_at = delivered_at

        # Return list of messages
        return message_ins_list

    # pylint: disable=R0911
    def store_message_res(self, message: Message) -> Optional[UUID]:
        """Store one Message."""
        # Validate message
        errors = validate_message(message, is_reply_message=True)
        if any(errors):
            log(ERROR, errors)
            return None

        res_metadata = message.metadata
        with self.lock:
            # Check if the Message it is replying to exists and is valid
            msg_ins_id = res_metadata.reply_to_message_id
            msg_ins = self.message_ins_store.get(UUID(msg_ins_id))

            # Ensure that dst_node_id of original Message matches the src_node_id of
            # reply Message.
            if (
                msg_ins
                and message
                and msg_ins.metadata.dst_node_id != res_metadata.src_node_id
            ):
                return None

            if msg_ins is None:
                log(
                    ERROR,
                    "Message with ID %s does not exist.",
                    msg_ins_id,
                )
                return None

            ins_metadata = msg_ins.metadata
            if ins_metadata.created_at + ins_metadata.ttl <= time.time():
                log(
                    ERROR,
                    "Failed to store Message: the message it is replying to "
                    "(with ID %s) has expired",
                    msg_ins_id,
                )
                return None

            # Fail if the Message TTL exceeds the
            # expiration time of the Message it replies to.
            # Condition: ins_metadata.created_at + ins_metadata.ttl ≥
            #            res_metadata.created_at + res_metadata.ttl
            # A small tolerance is introduced to account
            # for floating-point precision issues.
            max_allowed_ttl = (
                ins_metadata.created_at + ins_metadata.ttl - res_metadata.created_at
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

        # Validate run_id
        if res_metadata.run_id != ins_metadata.run_id:
            log(ERROR, "`metadata.run_id` is invalid")
            return None

        # Create message_id
        message_id = uuid4()

        # Store Message
        # pylint: disable-next=W0212
        message.metadata._message_id = str(message_id)  # type: ignore
        with self.lock:
            self.message_res_store[message_id] = message
            self.message_ins_id_to_message_res_id[UUID(msg_ins_id)] = message_id

        # Return the new message_id
        return message_id

    def get_message_res(self, message_ids: set[UUID]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        ret: dict[UUID, Message] = {}

        with self.lock:
            current = time.time()

            # Verify Message IDs
            ret = verify_message_ids(
                inquired_message_ids=message_ids,
                found_message_ins_dict=self.message_ins_store,
                current_time=current,
            )

            # Check node availability
            dst_node_ids = {
                self.message_ins_store[message_id].metadata.dst_node_id
                for message_id in message_ids
            }
            tmp_ret_dict = check_node_availability_for_in_message(
                inquired_in_message_ids=message_ids,
                found_in_message_dict=self.message_ins_store,
                node_id_to_online_until={
                    node_id: self.node_ids[node_id][0] for node_id in dst_node_ids
                },
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Find all reply Messages
            message_res_found: list[Message] = []
            for message_id in message_ids:
                # If Message exists and is not delivered, add it to the list
                if message_res_id := self.message_ins_id_to_message_res_id.get(
                    message_id
                ):
                    message_res = self.message_res_store[message_res_id]
                    if message_res.metadata.delivered_at == "":
                        message_res_found.append(message_res)
            tmp_ret_dict = verify_found_message_replies(
                inquired_message_ids=message_ids,
                found_message_ins_dict=self.message_ins_store,
                found_message_res_list=message_res_found,
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Mark existing reply Messages to be returned as delivered
            delivered_at = now().isoformat()
            for message_res in message_res_found:
                message_res.metadata.delivered_at = delivered_at

        return list(ret.values())

    def delete_messages(self, message_ins_ids: set[UUID]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        if not message_ins_ids:
            return

        with self.lock:
            for message_id in message_ins_ids:
                # Delete Messages
                if message_id in self.message_ins_store:
                    del self.message_ins_store[message_id]
                # Delete Message replies
                if message_id in self.message_ins_id_to_message_res_id:
                    message_res_id = self.message_ins_id_to_message_res_id.pop(
                        message_id
                    )
                    del self.message_res_store[message_res_id]

    def get_message_ids_from_run_id(self, run_id: int) -> set[UUID]:
        """Get all instruction Message IDs for the given run_id."""
        message_id_list: set[UUID] = set()
        with self.lock:
            for message_id, message in self.message_ins_store.items():
                if message.metadata.run_id == run_id:
                    message_id_list.add(message_id)

        return message_id_list

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store.

        This includes delivered but not yet deleted.
        """
        return len(self.message_ins_store)

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store.

        This includes delivered but not yet deleted.
        """
        return len(self.message_res_store)

    def create_node(self, ping_interval: float) -> int:
        """Create, store in the link state, and return `node_id`."""
        # Sample a random int64 as node_id
        node_id = generate_rand_int_from_bytes(
            NODE_ID_NUM_BYTES, exclude=[SUPERLINK_NODE_ID, 0]
        )

        with self.lock:
            if node_id in self.node_ids:
                log(ERROR, "Unexpected node registration failure.")
                return 0

            # Mark the node online util time.time() + ping_interval
            self.node_ids[node_id] = (time.time() + ping_interval, ping_interval)
            return node_id

    def delete_node(self, node_id: int) -> None:
        """Delete a node."""
        with self.lock:
            if node_id not in self.node_ids:
                raise ValueError(f"Node {node_id} not found")

            # Remove node ID <> public key mappings
            if pk := self.node_id_to_public_key.pop(node_id, None):
                del self.public_key_to_node_id[pk]

            del self.node_ids[node_id]

    def get_nodes(self, run_id: int) -> set[int]:
        """Return all available nodes.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        with self.lock:
            if run_id not in self.run_ids:
                return set()
            current_time = time.time()
            return {
                node_id
                for node_id, (online_until, _) in self.node_ids.items()
                if online_until > current_time
            }

    def set_node_public_key(self, node_id: int, public_key: bytes) -> None:
        """Set `public_key` for the specified `node_id`."""
        with self.lock:
            if node_id not in self.node_ids:
                raise ValueError(f"Node {node_id} not found")

            if public_key in self.public_key_to_node_id:
                raise ValueError("Public key already in use")

            self.public_key_to_node_id[public_key] = node_id
            self.node_id_to_public_key[node_id] = public_key

    def get_node_public_key(self, node_id: int) -> Optional[bytes]:
        """Get `public_key` for the specified `node_id`."""
        with self.lock:
            if node_id not in self.node_ids:
                raise ValueError(f"Node {node_id} not found")

            return self.node_id_to_public_key.get(node_id)

    def get_node_id(self, node_public_key: bytes) -> Optional[int]:
        """Retrieve stored `node_id` filtered by `node_public_keys`."""
        return self.public_key_to_node_id.get(node_public_key)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_run(
        self,
        fab_id: Optional[str],
        fab_version: Optional[str],
        fab_hash: Optional[str],
        override_config: UserConfig,
        federation_options: ConfigRecord,
    ) -> int:
        """Create a new run for the specified `fab_hash`."""
        # Sample a random int64 as run_id
        with self.lock:
            run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)

            if run_id not in self.run_ids:
                run_record = RunRecord(
                    run=Run(
                        run_id=run_id,
                        fab_id=fab_id if fab_id else "",
                        fab_version=fab_version if fab_version else "",
                        fab_hash=fab_hash if fab_hash else "",
                        override_config=override_config,
                        pending_at=now().isoformat(),
                        starting_at="",
                        running_at="",
                        finished_at="",
                        status=RunStatus(
                            status=Status.PENDING,
                            sub_status="",
                            details="",
                        ),
                    ),
                )
                self.run_ids[run_id] = run_record

                # Record federation options. Leave empty if not passed
                self.federation_options[run_id] = federation_options
                return run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def clear_supernode_auth_keys(self) -> None:
        """Clear stored `node_public_keys` in the link state if any."""
        with self.lock:
            self.node_public_keys.clear()

    def store_node_public_keys(self, public_keys: set[bytes]) -> None:
        """Store a set of `node_public_keys` in the link state."""
        with self.lock:
            self.node_public_keys.update(public_keys)

    def store_node_public_key(self, public_key: bytes) -> None:
        """Store a `node_public_key` in the link state."""
        with self.lock:
            self.node_public_keys.add(public_key)

    def get_node_public_keys(self) -> set[bytes]:
        """Retrieve all currently stored `node_public_keys` as a set."""
        with self.lock:
            return self.node_public_keys.copy()

    def get_run_ids(self) -> set[int]:
        """Retrieve all run IDs."""
        with self.lock:
            return set(self.run_ids.keys())

    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve information about the run with the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return None
            return self.run_ids[run_id].run

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        with self.lock:
            return {
                run_id: self.run_ids[run_id].run.status
                for run_id in set(run_ids)
                if run_id in self.run_ids
            }

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        with self.lock:
            # Check if the run_id exists
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return False

            # Check if the status transition is valid
            current_status = self.run_ids[run_id].run.status
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
            run_record = self.run_ids[run_id]
            if new_status.status == Status.STARTING:
                run_record.run.starting_at = now().isoformat()
            elif new_status.status == Status.RUNNING:
                run_record.run.running_at = now().isoformat()
            elif new_status.status == Status.FINISHED:
                run_record.run.finished_at = now().isoformat()
            run_record.run.status = new_status
            return True

    def get_pending_run_id(self) -> Optional[int]:
        """Get the `run_id` of a run with `Status.PENDING` status, if any."""
        pending_run_id = None

        # Loop through all registered runs
        for run_id, run_rec in self.run_ids.items():
            # Break once a pending run is found
            if run_rec.run.status.status == Status.PENDING:
                pending_run_id = run_id
                break

        return pending_run_id

    def get_federation_options(self, run_id: int) -> Optional[ConfigRecord]:
        """Retrieve the federation options for the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return None
            return self.federation_options[run_id]

    def acknowledge_ping(self, node_id: int, ping_interval: float) -> bool:
        """Acknowledge a ping received from a node, serving as a heartbeat.

        It allows for one missed ping (in a PING_PATIENCE * ping_interval) before
        marking the node as offline, where PING_PATIENCE = 2 in default.
        """
        with self.lock:
            if node_id in self.node_ids:
                self.node_ids[node_id] = (
                    time.time() + PING_PATIENCE * ping_interval,
                    ping_interval,
                )
                return True
        return False

    def get_serverapp_context(self, run_id: int) -> Optional[Context]:
        """Get the context for the specified `run_id`."""
        return self.contexts.get(run_id)

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        self.contexts[run_id] = context

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the serverapp logs for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        run = self.run_ids[run_id]
        with run.log_lock:
            run.logs.append((now().timestamp(), log_message))

    def get_serverapp_log(
        self, run_id: int, after_timestamp: Optional[float]
    ) -> tuple[str, float]:
        """Get the serverapp logs for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        run = self.run_ids[run_id]
        if after_timestamp is None:
            after_timestamp = 0.0
        with run.log_lock:
            # Find the index where the timestamp would be inserted
            index = bisect_right(run.logs, (after_timestamp, ""))
            latest_timestamp = run.logs[-1][0] if index < len(run.logs) else 0.0
            return "".join(log for _, log in run.logs[index:]), latest_timestamp

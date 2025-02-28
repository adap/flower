# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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

from flwr.common import Context, Message, Metadata, log, now
from flwr.common.constant import (
    MESSAGE_TTL_TOLERANCE,
    NODE_ID_NUM_BYTES,
    RUN_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.record import ConfigsRecord
from flwr.common.typing import Run, RunStatus, UserConfig
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.utils import validate_message, validate_task_ins_or_res

from .utils import (
    check_ttl_exceeded,
    create_message_error_expired_result_message,
    create_message_error_pending_result_message,
    create_message_error_unavailable_ins_message,
    generate_rand_int_from_bytes,
    has_valid_sub_status,
    is_valid_transition,
    message_ttl_has_expired,
    verify_found_taskres,
    verify_taskins_ids,
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
        self.federation_options: dict[int, ConfigsRecord] = {}
        self.task_ins_store: dict[UUID, TaskIns] = {}
        self.task_res_store: dict[UUID, TaskRes] = {}
        self.task_ins_id_to_task_res_id: dict[UUID, UUID] = {}

        self.message_ins_store: dict[UUID, Message] = {}
        self.message_res_store: dict[UUID, Message] = {}
        self.message_ins_to_message_res: dict[UUID, UUID] = {}
        self.delivered_messages: dict[UUID, Metadata] = {}
        # node-id to message-id mapping (for fast `get_message_ins` processing)
        self.dst_node_id_to_message_id_mapping: dict[int, list[UUID]] = {}

        self.node_public_keys: set[bytes] = set()

        self.lock = threading.RLock()

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""
        # Validate task
        errors = validate_task_ins_or_res(task_ins)
        if any(errors):
            log(ERROR, errors)
            return None
        # Validate run_id
        if task_ins.run_id not in self.run_ids:
            log(ERROR, "Invalid run ID for TaskIns: %s", task_ins.run_id)
            return None
        # Validate source node ID
        if task_ins.task.producer.node_id != SUPERLINK_NODE_ID:
            log(
                ERROR,
                "Invalid source node ID for TaskIns: %s",
                task_ins.task.producer.node_id,
            )
            return None
        # Validate destination node ID
        if task_ins.task.consumer.node_id not in self.node_ids:
            log(
                ERROR,
                "Invalid destination node ID for TaskIns: %s",
                task_ins.task.consumer.node_id,
            )
            return None

        # Create task_id
        task_id = uuid4()

        # Store TaskIns
        task_ins.task_id = str(task_id)
        with self.lock:
            self.task_ins_store[task_id] = task_ins

        # Return the new task_id
        return task_id

    def store_message_ins(self, message: Message) -> Optional[UUID]:
        """Store one Message from ServerAppIo."""
        # Validate message
        errors = validate_message(message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None
        metadata = message.metadata
        # Validate run_id
        if metadata.run_id not in self.run_ids:
            log(ERROR, "Invalid run ID for Message: %s", metadata.run_id)
            return None
        # Validate source node ID
        if metadata.src_node_id != SUPERLINK_NODE_ID:
            log(ERROR, "Invalid source node ID for Message: %s", metadata.src_node_id)
            return None
        # Validate destination node ID
        if metadata.dst_node_id not in self.node_ids:
            log(
                ERROR,
                "Invalid destination node ID for TaskIns: %s",
                metadata.dst_node_id,
            )
            return None

        # Create message_id
        message_id = uuid4()

        # Store Message
        message.metadata._message_id = str(message_id)  # type: ignore  # pylint: disable=W0212
        with self.lock:
            self.message_ins_store[message_id] = message

            # Record in mapping
            if metadata.dst_node_id in self.dst_node_id_to_message_id_mapping:
                self.dst_node_id_to_message_id_mapping[metadata.dst_node_id].append(
                    message_id
                )
            else:
                self.dst_node_id_to_message_id_mapping[metadata.dst_node_id] = [
                    message_id
                ]

        # Return the new message_id
        return message_id

    def get_task_ins(self, node_id: int, limit: Optional[int]) -> list[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find TaskIns for node_id that were not delivered yet
        task_ins_list: list[TaskIns] = []
        current_time = time.time()
        with self.lock:
            for _, task_ins in self.task_ins_store.items():
                if (
                    task_ins.task.consumer.node_id == node_id
                    and task_ins.task.delivered_at == ""
                    and task_ins.task.created_at + task_ins.task.ttl > current_time
                ):
                    task_ins_list.append(task_ins)
                if limit and len(task_ins_list) == limit:
                    break

        # Mark all of them as delivered
        delivered_at = now().isoformat()
        for task_ins in task_ins_list:
            task_ins.task.delivered_at = delivered_at

        # Return TaskIns
        return task_ins_list

    def get_message_ins(self, node_id: int, limit: Optional[int]) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        if node_id == SUPERLINK_NODE_ID:
            raise AssertionError(f"`node_id` must be != {SUPERLINK_NODE_ID}")

        # Find Messages for node_id
        message_list: list[Message] = []
        with self.lock:
            # Get all UUIDs of messages to be pulled by `node_id` node
            all_message_ids = self.dst_node_id_to_message_id_mapping.get(node_id, [])

            # If there are Messages associated to this node
            if all_message_ids:
                # Take at most `limit` message ids
                message_ids = all_message_ids[:limit] if limit else all_message_ids

                # Extract from Message store
                message_list = [
                    self.message_ins_store.pop(msg_id) for msg_id in message_ids
                ]

                # Remove message_ids from mapping
                self.dst_node_id_to_message_id_mapping[node_id] = all_message_ids[
                    limit:
                ]
                if len(self.dst_node_id_to_message_id_mapping[node_id]) == 0:
                    # Remove node entry if no message_ids left
                    del self.dst_node_id_to_message_id_mapping[node_id]

                # Record metadata of extracted messages
                for msg in message_list:
                    self.delivered_messages[UUID(msg.metadata.message_id)] = (
                        msg.metadata
                    )

        # Return Messages
        return message_list

    # pylint: disable=R0911
    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes."""
        # Validate task
        errors = validate_task_ins_or_res(task_res)
        if any(errors):
            log(ERROR, errors)
            return None

        with self.lock:
            # Check if the TaskIns it is replying to exists and is valid
            task_ins_id = task_res.task.ancestry[0]
            task_ins = self.task_ins_store.get(UUID(task_ins_id))

            # Ensure that the consumer_id of taskIns matches the producer_id of taskRes.
            if (
                task_ins
                and task_res
                and task_ins.task.consumer.node_id != task_res.task.producer.node_id
            ):
                return None

            if task_ins is None:
                log(ERROR, "TaskIns with task_id %s does not exist.", task_ins_id)
                return None

            if task_ins.task.created_at + task_ins.task.ttl <= time.time():
                log(
                    ERROR,
                    "Failed to store TaskRes: TaskIns with task_id %s has expired.",
                    task_ins_id,
                )
                return None

            # Fail if the TaskRes TTL exceeds the
            # expiration time of the TaskIns it replies to.
            # Condition: TaskIns.created_at + TaskIns.ttl ≥
            #            TaskRes.created_at + TaskRes.ttl
            # A small tolerance is introduced to account
            # for floating-point precision issues.
            max_allowed_ttl = (
                task_ins.task.created_at + task_ins.task.ttl - task_res.task.created_at
            )
            if task_res.task.ttl and (
                task_res.task.ttl - max_allowed_ttl > MESSAGE_TTL_TOLERANCE
            ):
                log(
                    WARNING,
                    "Received TaskRes with TTL %.2f "
                    "exceeding the allowed maximum TTL %.2f.",
                    task_res.task.ttl,
                    max_allowed_ttl,
                )
                return None

        # Validate run_id
        if task_res.run_id not in self.run_ids:
            log(ERROR, "`run_id` is invalid")
            return None

        # Create task_id
        task_id = uuid4()

        # Store TaskRes
        task_res.task_id = str(task_id)
        with self.lock:
            self.task_res_store[task_id] = task_res
            self.task_ins_id_to_task_res_id[UUID(task_ins_id)] = task_id

        # Return the new task_id
        return task_id

    def store_message_res(self, message: Message) -> Optional[UUID]:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=True)
        if any(errors):
            log(ERROR, errors)
            return None

        # Validate run_id
        if message.metadata.run_id not in self.run_ids:
            log(ERROR, "`run_id` is invalid")
            return None

        with self.lock:
            res_metadata = message.metadata
            # Check if message it is replying to exists and is valid
            # Find metadata of Message the `message` is the reply of
            if ins_metadata := self.delivered_messages.get(
                UUID(res_metadata.reply_to_message)
            ):
                # Ensure destination and source node_ids match
                if ins_metadata.dst_node_id != res_metadata.src_node_id:
                    log(
                        WARNING,
                        "Mismatch between source and destination node_ids in "
                        "received reply Message.",
                    )
                    return None
            else:
                # ins_metadata not found
                return None

        if ins_metadata.created_at + ins_metadata.ttl <= time.time():
            log(
                ERROR,
                "Failed to store Message reply: Original Message with message_id %s "
                " has expired.",
                ins_metadata.message_id,
            )
            return None

        if check_ttl_exceeded(ins_metadata, res_metadata):
            return None

        # Create message_id
        message_id = uuid4()

        # Store Message
        message.metadata._message_id = str(message_id)  # type: ignore   # pylint: disable=W0212
        with self.lock:
            self.message_res_store[message_id] = message
            self.message_ins_to_message_res[UUID(ins_metadata.message_id)] = message_id

        # Return the new message_id
        return message_id

    def get_task_res(self, task_ids: set[UUID]) -> list[TaskRes]:
        """Get TaskRes for the given TaskIns IDs."""
        ret: dict[UUID, TaskRes] = {}

        with self.lock:
            current = time.time()

            # Verify TaskIns IDs
            ret = verify_taskins_ids(
                inquired_taskins_ids=task_ids,
                found_taskins_dict=self.task_ins_store,
                current_time=current,
            )

            # Find all TaskRes
            task_res_found: list[TaskRes] = []
            for task_id in task_ids:
                # If TaskRes exists and is not delivered, add it to the list
                if task_res_id := self.task_ins_id_to_task_res_id.get(task_id):
                    task_res = self.task_res_store[task_res_id]
                    if task_res.task.delivered_at == "":
                        task_res_found.append(task_res)
            tmp_ret_dict = verify_found_taskres(
                inquired_taskins_ids=task_ids,
                found_taskins_dict=self.task_ins_store,
                found_taskres_list=task_res_found,
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Mark existing TaskRes to be returned as delivered
            delivered_at = now().isoformat()
            for task_res in task_res_found:
                task_res.task.delivered_at = delivered_at

        return list(ret.values())

    def get_message_res(self, message_ids: set[UUID]) -> list[Message]:
        """Get reply Message for the given Message IDs."""
        reply_list: list[Message] = []

        with self.lock:

            current_time = time.time()

            # Check first if reply Message arrived
            # If not, check if original Message had been pulled
            # If not, check if the original Message was ever recorded
            for msg_id in message_ids:

                # Has the reply message arrived ?
                if msg_res_id := self.message_ins_to_message_res.pop(msg_id, None):
                    #   expired TTL ? -> return Error Message
                    #   else -> Return actual Reply Message
                    msg_res = self.message_res_store.pop(msg_res_id)
                    if message_ttl_has_expired(
                        msg_res.metadata, current_time=current_time
                    ):
                        in_metadata = self.delivered_messages.pop(msg_id)
                        reply_list.append(
                            create_message_error_expired_result_message(in_metadata)
                        )
                    else:
                        del self.delivered_messages[msg_id]
                        reply_list.append(msg_res)

                else:
                    # Has the message being pulled by a SuperNode?
                    if msg_id in self.delivered_messages:
                        # if TTL expired -> Return TTL expired Error Message
                        # if TTL is still valid -> Return Error Message (awaiting reply)
                        msg_ins_metadata = self.delivered_messages[msg_id]
                        if message_ttl_has_expired(
                            msg_ins_metadata, current_time=current_time
                        ):
                            del self.delivered_messages[msg_id]
                            reply_list.append(
                                create_message_error_unavailable_ins_message(
                                    msg_ins_metadata
                                )
                            )
                        else:
                            reply_list.append(
                                create_message_error_pending_result_message(
                                    msg_ins_metadata
                                )
                            )

                    # Was the message ever pushed to the SuperLink by the ServerAppIO?
                    elif msg_id in self.message_ins_store:
                        # if TTL expired -> Return TTL expired Error Message
                        # if TTL valid -> Return Error Message (message not yet pulled)
                        msg_ins_metadata = self.message_ins_store[msg_id].metadata
                        if message_ttl_has_expired(
                            msg_ins_metadata, current_time=current_time
                        ):
                            # Remove message and reference in node-to-message mapping
                            msg_ins = self.message_ins_store.pop(msg_id)
                            self.dst_node_id_to_message_id_mapping[
                                msg_ins.metadata.dst_node_id
                            ].remove(msg_id)
                            reply_list.append(
                                create_message_error_unavailable_ins_message(
                                    msg_ins_metadata
                                )
                            )
                        else:
                            reply_list.append(
                                create_message_error_pending_result_message(
                                    msg_ins_metadata
                                )
                            )
                    else:
                        # msg_id isn't recognized
                        meta = Metadata(
                            run_id=0,
                            message_id="",
                            src_node_id=SUPERLINK_NODE_ID,
                            dst_node_id=SUPERLINK_NODE_ID,
                            reply_to_message="",
                            group_id="",
                            ttl=SUPERLINK_NODE_ID,
                            message_type="",
                        )
                        meta._created_at = current_time  # type: ignore
                        reply_list.append(
                            create_message_error_unavailable_ins_message(meta)
                        )

        return reply_list

    def delete_tasks(self, task_ins_ids: set[UUID]) -> None:
        """Delete TaskIns/TaskRes pairs based on provided TaskIns IDs."""
        if not task_ins_ids:
            return

        with self.lock:
            for task_id in task_ins_ids:
                # Delete TaskIns
                if task_id in self.task_ins_store:
                    del self.task_ins_store[task_id]
                # Delete TaskRes
                if task_id in self.task_ins_id_to_task_res_id:
                    task_res_id = self.task_ins_id_to_task_res_id.pop(task_id)
                    del self.task_res_store[task_res_id]

    def get_task_ids_from_run_id(self, run_id: int) -> set[UUID]:
        """Get all TaskIns IDs for the given run_id."""
        task_id_list: set[UUID] = set()
        with self.lock:
            for task_id, task_ins in self.task_ins_store.items():
                if task_ins.run_id == run_id:
                    task_id_list.add(task_id)

        return task_id_list

    def num_task_ins(self) -> int:
        """Calculate the number of task_ins in store.

        This includes delivered but not yet deleted task_ins.
        """
        return len(self.task_ins_store)

    def num_message_ins(self) -> int:
        """Calculate the number of Messages awaiting a reply."""
        return len(self.message_ins_store) + len(self.delivered_messages)

    def num_task_res(self) -> int:
        """Calculate the number of task_res in store.

        This includes delivered but not yet deleted task_res.
        """
        return len(self.task_res_store)

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store."""
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
        federation_options: ConfigsRecord,
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

    def get_federation_options(self, run_id: int) -> Optional[ConfigsRecord]:
        """Retrieve the federation options for the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return None
            return self.federation_options[run_id]

    def acknowledge_ping(self, node_id: int, ping_interval: float) -> bool:
        """Acknowledge a ping received from a node, serving as a heartbeat."""
        with self.lock:
            if node_id in self.node_ids:
                self.node_ids[node_id] = (time.time() + ping_interval, ping_interval)
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

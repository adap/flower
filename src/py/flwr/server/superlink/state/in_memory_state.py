# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""In-memory State implementation."""


import os
import threading
import time
from logging import ERROR
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from flwr.common import log, now
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.state.state import State
from flwr.server.utils import validate_task_ins_or_res

from .utils import make_node_unavailable_taskres


class InMemoryState(State):  # pylint: disable=R0902,R0904
    """In-memory State implementation."""

    def __init__(self) -> None:

        # Map node_id to (online_until, ping_interval)
        self.node_ids: Dict[int, Tuple[float, float]] = {}
        self.public_key_to_node_id: Dict[bytes, int] = {}

        # Map run_id to (fab_id, fab_version)
        self.run_ids: Dict[int, Tuple[str, str]] = {}
        self.task_ins_store: Dict[UUID, TaskIns] = {}
        self.task_res_store: Dict[UUID, TaskRes] = {}

        self.client_public_keys: Set[bytes] = set()
        self.server_public_key: Optional[bytes] = None
        self.server_private_key: Optional[bytes] = None

        self.lock = threading.Lock()

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""
        # Validate task
        errors = validate_task_ins_or_res(task_ins)
        if any(errors):
            log(ERROR, errors)
            return None
        # Validate run_id
        if task_ins.run_id not in self.run_ids:
            log(ERROR, "`run_id` is invalid")
            return None

        # Create task_id
        task_id = uuid4()

        # Store TaskIns
        task_ins.task_id = str(task_id)
        with self.lock:
            self.task_ins_store[task_id] = task_ins

        # Return the new task_id
        return task_id

    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find TaskIns for node_id that were not delivered yet
        task_ins_list: List[TaskIns] = []
        with self.lock:
            for _, task_ins in self.task_ins_store.items():
                # pylint: disable=too-many-boolean-expressions
                if (
                    node_id is not None  # Not anonymous
                    and task_ins.task.consumer.anonymous is False
                    and task_ins.task.consumer.node_id == node_id
                    and task_ins.task.delivered_at == ""
                ) or (
                    node_id is None  # Anonymous
                    and task_ins.task.consumer.anonymous is True
                    and task_ins.task.consumer.node_id == 0
                    and task_ins.task.delivered_at == ""
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

    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes."""
        # Validate task
        errors = validate_task_ins_or_res(task_res)
        if any(errors):
            log(ERROR, errors)
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

        # Return the new task_id
        return task_id

    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get all TaskRes that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        with self.lock:
            # Find TaskRes that were not delivered yet
            task_res_list: List[TaskRes] = []
            replied_task_ids: Set[UUID] = set()
            for _, task_res in self.task_res_store.items():
                reply_to = UUID(task_res.task.ancestry[0])
                if reply_to in task_ids and task_res.task.delivered_at == "":
                    task_res_list.append(task_res)
                    replied_task_ids.add(reply_to)
                if limit and len(task_res_list) == limit:
                    break

            # Check if the node is offline
            for task_id in task_ids - replied_task_ids:
                if limit and len(task_res_list) == limit:
                    break
                task_ins = self.task_ins_store.get(task_id)
                if task_ins is None:
                    continue
                node_id = task_ins.task.consumer.node_id
                online_until, _ = self.node_ids[node_id]
                # Generate a TaskRes containing an error reply if the node is offline.
                if online_until < time.time():
                    err_taskres = make_node_unavailable_taskres(
                        ref_taskins=task_ins,
                    )
                    self.task_res_store[UUID(err_taskres.task_id)] = err_taskres
                    task_res_list.append(err_taskres)

            # Mark all of them as delivered
            delivered_at = now().isoformat()
            for task_res in task_res_list:
                task_res.task.delivered_at = delivered_at

            # Return TaskRes
            return task_res_list

    def delete_tasks(self, task_ids: Set[UUID]) -> None:
        """Delete all delivered TaskIns/TaskRes pairs."""
        task_ins_to_be_deleted: Set[UUID] = set()
        task_res_to_be_deleted: Set[UUID] = set()

        with self.lock:
            for task_ins_id in task_ids:
                # Find the task_id of the matching task_res
                for task_res_id, task_res in self.task_res_store.items():
                    if UUID(task_res.task.ancestry[0]) != task_ins_id:
                        continue
                    if task_res.task.delivered_at == "":
                        continue

                    task_ins_to_be_deleted.add(task_ins_id)
                    task_res_to_be_deleted.add(task_res_id)

            for task_id in task_ins_to_be_deleted:
                del self.task_ins_store[task_id]
            for task_id in task_res_to_be_deleted:
                del self.task_res_store[task_id]

    def num_task_ins(self) -> int:
        """Calculate the number of task_ins in store.

        This includes delivered but not yet deleted task_ins.
        """
        return len(self.task_ins_store)

    def num_task_res(self) -> int:
        """Calculate the number of task_res in store.

        This includes delivered but not yet deleted task_res.
        """
        return len(self.task_res_store)

    def create_node(
        self, ping_interval: float, public_key: Optional[bytes] = None
    ) -> int:
        """Create, store in state, and return `node_id`."""
        # Sample a random int64 as node_id
        node_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

        with self.lock:
            if node_id in self.node_ids:
                log(ERROR, "Unexpected node registration failure.")
                return 0

            if public_key is not None:
                if (
                    public_key in self.public_key_to_node_id
                    or self.public_key_to_node_id.get(public_key) == node_id
                ):
                    log(ERROR, "Unexpected node registration failure.")
                    return 0

                self.public_key_to_node_id[public_key] = node_id

            self.node_ids[node_id] = (time.time() + ping_interval, ping_interval)
            return node_id

    def delete_node(self, node_id: int, public_key: Optional[bytes] = None) -> None:
        """Delete a client node."""
        with self.lock:
            if node_id not in self.node_ids:
                raise ValueError(f"Node {node_id} not found")

            if public_key is not None:
                if (
                    public_key not in self.public_key_to_node_id
                    and self.public_key_to_node_id.get(public_key) != node_id
                ):
                    raise ValueError("Public key or node_id not found")

                del self.public_key_to_node_id[public_key]

            del self.node_ids[node_id]

    def get_nodes(self, run_id: int) -> Set[int]:
        """Return all available client nodes.

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

    def get_node_id(self, client_public_key: bytes) -> Optional[int]:
        """Retrieve stored `node_id` filtered by `client_public_keys`."""
        return self.public_key_to_node_id.get(client_public_key)

    def create_run(self, fab_id: str, fab_version: str) -> int:
        """Create a new run for the specified `fab_id` and `fab_version`."""
        # Sample a random int64 as run_id
        with self.lock:
            run_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

            if run_id not in self.run_ids:
                self.run_ids[run_id] = (fab_id, fab_version)
                return run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def store_server_private_public_key(
        self, private_key: bytes, public_key: bytes
    ) -> None:
        """Store `server_private_key` and `server_public_key` in state."""
        with self.lock:
            if self.server_private_key is None and self.server_public_key is None:
                self.server_private_key = private_key
                self.server_public_key = public_key
            else:
                raise RuntimeError("Server private and public key already set")

    def get_server_private_key(self) -> Optional[bytes]:
        """Retrieve `server_private_key` in urlsafe bytes."""
        return self.server_private_key

    def get_server_public_key(self) -> Optional[bytes]:
        """Retrieve `server_public_key` in urlsafe bytes."""
        return self.server_public_key

    def store_client_public_keys(self, public_keys: Set[bytes]) -> None:
        """Store a set of `client_public_keys` in state."""
        with self.lock:
            self.client_public_keys = public_keys

    def store_client_public_key(self, public_key: bytes) -> None:
        """Store a `client_public_key` in state."""
        with self.lock:
            self.client_public_keys.add(public_key)

    def get_client_public_keys(self) -> Set[bytes]:
        """Retrieve all currently stored `client_public_keys` as a set."""
        return self.client_public_keys

    def get_run(self, run_id: int) -> Tuple[int, str, str]:
        """Retrieve information about the run with the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return 0, "", ""
            return run_id, *self.run_ids[run_id]

    def acknowledge_ping(self, node_id: int, ping_interval: float) -> bool:
        """Acknowledge a ping received from a node, serving as a heartbeat."""
        with self.lock:
            if node_id in self.node_ids:
                self.node_ids[node_id] = (time.time() + ping_interval, ping_interval)
                return True
        return False

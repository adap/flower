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
from datetime import datetime, timedelta
from logging import ERROR
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from flwr.common import log, now
from flwr.proto.task_pb2 import TaskIns, TaskRes
from flwr.server.state.state import State
from flwr.server.utils import validate_task_ins_or_res


class InMemoryState(State):
    """In-memory State implementation."""

    def __init__(self) -> None:
        self.node_ids: Set[int] = set()
        self.workload_ids: Set[int] = set()
        self.task_ins_store: Dict[UUID, TaskIns] = {}
        self.task_res_store: Dict[UUID, TaskRes] = {}

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""
        # Validate task
        errors = validate_task_ins_or_res(task_ins)
        if any(errors):
            log(ERROR, errors)
            return None
        # Validate workload_id
        if task_ins.workload_id not in self.workload_ids:
            log(ERROR, "`workload_id` is invalid")
            return None

        # Create task_id, created_at and ttl
        task_id = uuid4()
        created_at: datetime = now()
        ttl: datetime = created_at + timedelta(hours=24)

        # Store TaskIns
        task_ins.task_id = str(task_id)
        task_ins.task.created_at = created_at.isoformat()
        task_ins.task.ttl = ttl.isoformat()
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

        # Validate workload_id
        if task_res.workload_id not in self.workload_ids:
            log(ERROR, "`workload_id` is invalid")
            return None

        # Create task_id, created_at and ttl
        task_id = uuid4()
        created_at: datetime = now()
        ttl: datetime = created_at + timedelta(hours=24)

        # Store TaskRes
        task_res.task_id = str(task_id)
        task_res.task.created_at = created_at.isoformat()
        task_res.task.ttl = ttl.isoformat()
        self.task_res_store[task_id] = task_res

        # Return the new task_id
        return task_id

    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get all TaskRes that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find TaskRes that were not delivered yet
        task_res_list: List[TaskRes] = []
        for _, task_res in self.task_res_store.items():
            if (
                UUID(task_res.task.ancestry[0]) in task_ids
                and task_res.task.delivered_at == ""
            ):
                task_res_list.append(task_res)
            if limit and len(task_res_list) == limit:
                break

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

    def create_node(self) -> int:
        """Create, store in state, and return `node_id`."""
        # Sample a random int64 as node_id
        node_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

        if node_id not in self.node_ids:
            self.node_ids.add(node_id)
            return node_id
        log(ERROR, "Unexpected node registration failure.")
        return 0

    def delete_node(self, node_id: int) -> None:
        """Delete a client node."""
        if node_id not in self.node_ids:
            raise ValueError(f"Node {node_id} not found")
        self.node_ids.remove(node_id)

    def get_nodes(self, workload_id: int) -> Set[int]:
        """Return all available client nodes.

        Constraints
        -----------
        If the provided `workload_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        if workload_id not in self.workload_ids:
            return set()
        return self.node_ids

    def create_workload(self) -> int:
        """Create one workload."""
        # Sample a random int64 as workload_id
        workload_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

        if workload_id not in self.workload_ids:
            self.workload_ids.add(workload_id)
            return workload_id
        log(ERROR, "Unexpected workload creation failure.")
        return 0

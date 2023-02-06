# Copyright 2023 Adap GmbH. All Rights Reserved.
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


from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from flwr.proto.task_pb2 import TaskIns, TaskRes

from .state import State


class InMemoryState(State):
    """In-memory State implementation."""

    def __init__(self) -> None:
        self.node_ids: Set[int] = set()
        self.task_ins_store: Dict[UUID, TaskIns] = {}
        self.task_res_store: Dict[UUID, TaskRes] = {}

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""

        # Create and set task_id
        task_id = uuid4()
        task_ins.task_id = str(task_id)

        # Set created_at
        created_at: datetime = _now()
        ttl: datetime = created_at + timedelta(hours=24)

        # Store TaskIns
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
        delivered_at = _now().isoformat()
        for task_ins in task_ins_list:
            task_ins.task.delivered_at = delivered_at

        # Return TaskIns
        return task_ins_list

    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes."""

        # Create and set task_id
        task_id = uuid4()
        task_res.task_id = str(task_id)

        # Set created_at
        created_at: datetime = _now()
        ttl: datetime = created_at + timedelta(hours=24)

        # Store TaskRes
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
        delivered_at = _now().isoformat()
        for task_res in task_res_list:
            task_res.task.delivered_at = delivered_at

        # Return TaskRes
        return task_res_list

    def register_node(self, node_id: int) -> None:
        """Register a client node."""
        if node_id in self.node_ids:
            raise ValueError(f"Node {node_id} is already registered")
        self.node_ids.add(node_id)

    def unregister_node(self, node_id: int) -> None:
        """Unregister a client node."""
        if node_id not in self.node_ids:
            raise ValueError(f"Node {node_id} is not registered")
        self.node_ids.remove(node_id)

    def get_nodes(self) -> Set[int]:
        """Return all available client nodes."""
        return self.node_ids


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)

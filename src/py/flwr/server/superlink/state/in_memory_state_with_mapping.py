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
"""In-memory State with Mapping implementation."""

import random
from datetime import datetime, timedelta
from logging import ERROR
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from flwr.common import log, now
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611
from flwr.server.utils import validate_task_ins_or_res

from .in_memory_state import InMemoryState


class InMemoryStateWithMapping(InMemoryState):
    """In-memory State with Mapping implementation."""

    def __init__(self) -> None:
        self.task_ins_mapping: Dict[int, List[UUID]] = {}
        super().__init__()

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

        # Create task_id, created_at and ttl
        task_id = uuid4()
        created_at: datetime = now()
        ttl: datetime = created_at + timedelta(hours=24)

        # Store TaskIns
        task_ins.task_id = str(task_id)
        task_ins.task.created_at = created_at.isoformat()
        task_ins.task.ttl = ttl.isoformat()
        with self.lock:
            self.task_ins_store[task_id] = task_ins
            node_id = task_ins.task.consumer.node_id
            if node_id:
                # If not an annonymous node, let's construct or
                # update the node_id:task_id mapping
                if node_id in self.task_ins_mapping:
                    self.task_ins_mapping[node_id].append(task_id)
                else:
                    self.task_ins_mapping[node_id] = [task_id]

        # Return the new task_id
        return task_id

    def get_task_ins(self, node_id: Optional[int], limit: Optional[int]) -> List[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find TaskIns for node_id that were not delivered yet
        task_ins_list: List[TaskIns] = []
        with self.lock:
            num_to_return = self.num_task_ins()
            if num_to_return == 0:
                return task_ins_list
            if limit:
                num_to_return = min(num_to_return, limit)

            uuid = random.choice(list(self.task_ins_store.keys()))
            taskins = self.task_ins_store.pop(uuid)
            task_ins_list.append(taskins)

        # Mark all of them as delivered
        delivered_at = now().isoformat()
        for task_ins in task_ins_list:
            task_ins.task.delivered_at = delivered_at

        # Return TaskIns
        return task_ins_list

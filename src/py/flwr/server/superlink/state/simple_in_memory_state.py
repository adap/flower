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
"""In-memory State implementation."""


import random
from logging import WARN
from typing import List, Optional

from flwr.common import now
from flwr.common.logger import log
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .in_memory_state import InMemoryState
class SimpleInMemoryState(InMemoryState):  # pylint: disable=R0902,R0904
    """Simple In-memory State implementation."""


    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find TaskIns for node_id that were not delivered yet
        task_ins_list: List[TaskIns] = []

        with self.lock:
            # Get UUIDs of undelivered
            undelivered_uuids = self._get_undelivered_uuids(node_id)
            if len(undelivered_uuids) == 0:
                return task_ins_list
            if limit:
                num_to_return = min(len(undelivered_uuids), limit)
            while len(task_ins_list) < num_to_return:
                uuid = random.choice(self._get_undelivered_uuids(node_id))
                # Extract and mark as delivered
                task_ins = self.task_ins_store[uuid]
                task_ins.task.delivered_at = now().isoformat()
                task_ins_list.append(task_ins)

        # Return TaskIns
        return task_ins_list


    def _get_undelivered_uuids(self, node_id):
        if node_id is None:
                undelivered_uuids = [uuid for uuid, task_ins in self.task_ins_store.items() if (task_ins.task.delivered_at == "")]
        else:
            undelivered_uuids = [uuid for uuid, task_ins in self.task_ins_store.items() if (task_ins.task.delivered_at == "" and task_ins.task.consumer.node_id == node_id)]
        return undelivered_uuids 
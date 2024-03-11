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
from logging import WARN
from typing import List, Optional

from flwr.common import now
from flwr.common.logger import log
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .in_memory_state import InMemoryState


class SimpleInMemoryState(InMemoryState):
    """Variation of In-memory State (`InMemoryState`) better suited for simulation.

    In particular, `get_task_ins` relaxes the extraction of tasks by returning any N
    TaskIns. Ignoring the node-id field of a TaskIns is particularly beneficial when
    many thousands or millions of nodes are registered with the SuperLink.
    """

    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get `limit` TaskIns that have not been delivered yet.

        TaskIns are extracted from the store and returned regardless of their node_id
        assignment.
        """
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        if node_id:
            log(
                WARN,
                "%s won't return TaskIns of the specified node.",
                self.__class__.__name__,
            )

        # Find TaskIns for node_id that were not delivered yet
        task_ins_list: List[TaskIns] = []
        with self.lock:
            num_to_return = self.num_task_ins()
            if num_to_return == 0:
                return task_ins_list
            # ensure we don't return more tasks than what the state has
            if limit:
                num_to_return = min(num_to_return, limit)

            while len(task_ins_list) < num_to_return:
                uuid = random.choice(list(self.task_ins_store.keys()))
                taskins = self.task_ins_store.pop(uuid)
                task_ins_list.append(taskins)

        # Mark all of them as delivered
        delivered_at = now().isoformat()
        for task_ins in task_ins_list:
            task_ins.task.delivered_at = delivered_at

        # Return TaskIns
        return task_ins_list

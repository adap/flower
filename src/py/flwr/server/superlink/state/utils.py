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


import time
from uuid import uuid4

# pylint: disable=E0611
from flwr.proto.error_pb2 import Error
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes

# pylint: enable=E0611


NODE_UNAVAILABLE_ERROR_REASON = (
    "Error: Node Unavailable - The destination node is currently unavailable"
)


def make_node_unavailable_taskres(ref_taskins: TaskIns) -> TaskRes:
    """Create a TaskRes containing the node available error based on the reference
    TaskIns."""
    return TaskRes(
        task_id=str(uuid4()),
        group_id=ref_taskins.group_id,
        run_id=ref_taskins.run_id,
        task=Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=ref_taskins.task.producer.node_id, anonymous=False),
            created_at=time.time(),
            ttl=0,
            ancestry=[ref_taskins.task_id],
            task_type=ref_taskins.task.task_type,
            error=Error(code=3, reason=NODE_UNAVAILABLE_ERROR_REASON),
        ),
    )

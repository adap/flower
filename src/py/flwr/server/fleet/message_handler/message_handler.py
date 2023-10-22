# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Fleet API message handlers."""


from typing import List, Optional
from uuid import UUID

from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
    Reconnect,
)
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskIns, TaskRes
from flwr.server.state import State


def create_node(
    request: CreateNodeRequest,  # pylint: disable=unused-argument
    state: State,
) -> CreateNodeResponse:
    """."""
    # Create node
    node_id = state.create_node()
    return CreateNodeResponse(node=Node(node_id=node_id, anonymous=False))


def delete_node(request: DeleteNodeRequest, state: State) -> DeleteNodeResponse:
    """."""
    # Validate node_id
    if request.node.anonymous or request.node.node_id <= 0:
        return DeleteNodeResponse()

    # Update state
    state.delete_node(node_id=request.node.node_id)
    return DeleteNodeResponse()


def pull_task_ins(request: PullTaskInsRequest, state: State) -> PullTaskInsResponse:
    """Pull TaskIns handler."""
    # Get node_id if client node is not anonymous
    node = request.node  # pylint: disable=no-member
    node_id: Optional[int] = None if node.anonymous else node.node_id

    # Retrieve TaskIns from State
    task_ins_list: List[TaskIns] = state.get_task_ins(node_id=node_id, limit=1)

    # Build response
    response = PullTaskInsResponse(
        task_ins_list=task_ins_list,
    )
    return response


def push_task_res(request: PushTaskResRequest, state: State) -> PushTaskResResponse:
    """Push TaskRes handler."""
    # pylint: disable=no-member
    task_res: TaskRes = request.task_res_list[0]
    # pylint: enable=no-member

    # Store TaskRes in State
    task_id: Optional[UUID] = state.store_task_res(task_res=task_res)

    # Build response
    response = PushTaskResResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(task_id): 0},
    )
    return response

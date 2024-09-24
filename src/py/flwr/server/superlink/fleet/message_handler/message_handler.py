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
"""Fleet API message handlers."""


import time
from typing import Optional
from uuid import UUID

from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
    Reconnect,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.server.superlink.state import State


def create_node(
    request: CreateNodeRequest,  # pylint: disable=unused-argument
    state: State,
) -> CreateNodeResponse:
    """."""
    # Create node
    node_id = state.create_node(ping_interval=request.ping_interval)
    return CreateNodeResponse(node=Node(node_id=node_id, anonymous=False))


def delete_node(request: DeleteNodeRequest, state: State) -> DeleteNodeResponse:
    """."""
    # Validate node_id
    if request.node.anonymous or request.node.node_id == 0:
        return DeleteNodeResponse()

    # Update state
    state.delete_node(node_id=request.node.node_id)
    return DeleteNodeResponse()


def ping(
    request: PingRequest,  # pylint: disable=unused-argument
    state: State,  # pylint: disable=unused-argument
) -> PingResponse:
    """."""
    res = state.acknowledge_ping(request.node.node_id, request.ping_interval)
    return PingResponse(success=res)


def pull_task_ins(request: PullTaskInsRequest, state: State) -> PullTaskInsResponse:
    """Pull TaskIns handler."""
    # Get node_id if client node is not anonymous
    node = request.node  # pylint: disable=no-member
    node_id: Optional[int] = None if node.anonymous else node.node_id

    # Retrieve TaskIns from State
    task_ins_list: list[TaskIns] = state.get_task_ins(node_id=node_id, limit=1)

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

    # Set pushed_at (timestamp in seconds)
    task_res.task.pushed_at = time.time()

    # Store TaskRes in State
    task_id: Optional[UUID] = state.store_task_res(task_res=task_res)

    # Build response
    response = PushTaskResResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(task_id): 0},
    )
    return response


def get_run(
    request: GetRunRequest, state: State  # pylint: disable=W0613
) -> GetRunResponse:
    """Get run information."""
    run = state.get_run(request.run_id)

    if run is None:
        return GetRunResponse()

    return GetRunResponse(
        run=Run(
            run_id=run.run_id,
            fab_id=run.fab_id,
            fab_version=run.fab_version,
            override_config=user_config_to_proto(run.override_config),
            fab_hash=run.fab_hash,
        )
    )


def get_fab(
    request: GetFabRequest, ffs: Ffs  # pylint: disable=W0613
) -> GetFabResponse:
    """Get FAB."""
    if result := ffs.get(request.hash_str):
        fab = Fab(request.hash_str, result[0])
        return GetFabResponse(fab=fab_to_proto(fab))

    raise ValueError(f"Found no FAB with hash: {request.hash_str}")

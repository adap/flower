# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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


from typing import Optional
from uuid import UUID

from flwr.common import Message
from flwr.common.constant import Status
from flwr.common.serde import (
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    user_config_to_proto,
)
from flwr.common.typing import Fab, InvalidRunStatusException
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
    Reconnect,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.utils import check_abort


def create_node(
    request: CreateNodeRequest,  # pylint: disable=unused-argument
    state: LinkState,
) -> CreateNodeResponse:
    """."""
    # Create node
    node_id = state.create_node(ping_interval=request.ping_interval)
    return CreateNodeResponse(node=Node(node_id=node_id))


def delete_node(request: DeleteNodeRequest, state: LinkState) -> DeleteNodeResponse:
    """."""
    # Validate node_id
    if request.node.node_id == 0:  # i.e. unset `node_id`
        return DeleteNodeResponse()

    # Update state
    state.delete_node(node_id=request.node.node_id)
    return DeleteNodeResponse()


def ping(
    request: PingRequest,  # pylint: disable=unused-argument
    state: LinkState,  # pylint: disable=unused-argument
) -> PingResponse:
    """."""
    res = state.acknowledge_ping(request.node.node_id, request.ping_interval)
    return PingResponse(success=res)


def pull_messages(
    request: PullMessagesRequest, state: LinkState
) -> PullMessagesResponse:
    """Pull Messages handler."""
    # Get node_id if client node is not anonymous
    node = request.node  # pylint: disable=no-member
    node_id: int = node.node_id

    # Retrieve Message from State
    message_list: list[Message] = state.get_message_ins(node_id=node_id, limit=1)

    # Convert to Messages
    msg_proto = []
    for msg in message_list:
        msg_proto.append(message_to_proto(msg))

    return PullMessagesResponse(messages_list=msg_proto)


def push_messages(
    request: PushMessagesRequest, state: LinkState
) -> PushMessagesResponse:
    """Push Messages handler."""
    # Convert Message from proto
    msg = message_from_proto(message_proto=request.messages_list[0])

    # Abort if the run is not running
    abort_msg = check_abort(
        msg.metadata.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Store Message in State
    message_id: Optional[UUID] = state.store_message_res(message=msg)

    # Build response
    response = PushMessagesResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(message_id): 0},
    )
    return response


def get_run(request: GetRunRequest, state: LinkState) -> GetRunResponse:
    """Get run information."""
    run = state.get_run(request.run_id)

    if run is None:
        return GetRunResponse()

    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

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
    request: GetFabRequest, ffs: Ffs, state: LinkState  # pylint: disable=W0613
) -> GetFabResponse:
    """Get FAB."""
    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    if result := ffs.get(request.hash_str):
        fab = Fab(request.hash_str, result[0])
        return GetFabResponse(fab=fab_to_proto(fab))

    raise ValueError(f"Found no FAB with hash: {request.hash_str}")

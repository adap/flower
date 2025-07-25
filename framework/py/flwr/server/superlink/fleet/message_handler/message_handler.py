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

from logging import ERROR
from typing import Optional

from flwr.common import Message, log
from flwr.common.constant import Status
from flwr.common.inflatable import UnexpectedObjectContentError
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
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
    Reconnect,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.utils import check_abort
from flwr.supercore.ffs import Ffs
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStore
from flwr.supercore.object_store.utils import store_mapping_and_register_objects


def create_node(
    request: CreateNodeRequest,  # pylint: disable=unused-argument
    state: LinkState,
) -> CreateNodeResponse:
    """."""
    # Create node
    node_id = state.create_node(heartbeat_interval=request.heartbeat_interval)
    return CreateNodeResponse(node=Node(node_id=node_id))


def delete_node(request: DeleteNodeRequest, state: LinkState) -> DeleteNodeResponse:
    """."""
    # Validate node_id
    if request.node.node_id == 0:  # i.e. unset `node_id`
        return DeleteNodeResponse()

    # Update state
    state.delete_node(node_id=request.node.node_id)
    return DeleteNodeResponse()


def send_node_heartbeat(
    request: SendNodeHeartbeatRequest,  # pylint: disable=unused-argument
    state: LinkState,  # pylint: disable=unused-argument
) -> SendNodeHeartbeatResponse:
    """."""
    res = state.acknowledge_node_heartbeat(
        request.node.node_id, request.heartbeat_interval
    )
    return SendNodeHeartbeatResponse(success=res)


def pull_messages(
    request: PullMessagesRequest,
    state: LinkState,
    store: ObjectStore,
) -> PullMessagesResponse:
    """Pull Messages handler."""
    # Get node_id if client node is not anonymous
    node = request.node  # pylint: disable=no-member
    node_id: int = node.node_id

    # Retrieve Message from State
    message_list: list[Message] = state.get_message_ins(node_id=node_id, limit=1)

    # Convert to Messages
    msg_proto = []
    trees = []
    for msg in message_list:
        try:
            # Retrieve Message object tree from ObjectStore
            msg_object_id = msg.metadata.message_id
            obj_tree = store.get_object_tree(msg_object_id)

            # Add Message and its object tree to the response
            msg_proto.append(message_to_proto(msg))
            trees.append(obj_tree)
        except NoObjectInStoreError as e:
            log(ERROR, e.message)
            # Delete message ins from state
            state.delete_messages(message_ins_ids={msg_object_id})

    return PullMessagesResponse(messages_list=msg_proto, message_object_trees=trees)


def push_messages(
    request: PushMessagesRequest,
    state: LinkState,
    store: ObjectStore,
) -> PushMessagesResponse:
    """Push Messages handler."""
    # Convert Message from proto
    msg = message_from_proto(message_proto=request.messages_list[0])

    # Abort if the run is not running
    abort_msg = check_abort(
        msg.metadata.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Store Message in State
    message_id: Optional[str] = state.store_message_res(message=msg)

    # Store Message object to descendants mapping and preregister objects
    objects_to_push = store_mapping_and_register_objects(store, request=request)

    # Build response
    response = PushMessagesResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(message_id): 0},
        objects_to_push=objects_to_push,
    )
    return response


def get_run(
    request: GetRunRequest, state: LinkState, store: ObjectStore
) -> GetRunResponse:
    """Get run information."""
    run = state.get_run(request.run_id)

    if run is None:
        return GetRunResponse()

    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
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
    request: GetFabRequest, ffs: Ffs, state: LinkState, store: ObjectStore
) -> GetFabResponse:
    """Get FAB."""
    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    if result := ffs.get(request.hash_str):
        fab = Fab(request.hash_str, result[0])
        return GetFabResponse(fab=fab_to_proto(fab))

    raise ValueError(f"Found no FAB with hash: {request.hash_str}")


def push_object(
    request: PushObjectRequest, state: LinkState, store: ObjectStore
) -> PushObjectResponse:
    """Push Object."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    stored = False
    try:
        store.put(request.object_id, request.object_content)
        stored = True
    except (NoObjectInStoreError, ValueError) as e:
        log(ERROR, str(e))
    except UnexpectedObjectContentError as e:
        # Object content is not valid
        log(ERROR, str(e))
        raise
    return PushObjectResponse(stored=stored)


def pull_object(
    request: PullObjectRequest, state: LinkState, store: ObjectStore
) -> PullObjectResponse:
    """Pull Object."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Fetch from store
    content = store.get(request.object_id)
    if content is not None:
        object_available = content != b""
        return PullObjectResponse(
            object_found=True,
            object_available=object_available,
            object_content=content,
        )
    return PullObjectResponse(object_found=False, object_available=False)


def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
    state: LinkState,
    store: ObjectStore,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received handler."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Delete the message object
    store.delete(request.message_object_id)

    return ConfirmMessageReceivedResponse()

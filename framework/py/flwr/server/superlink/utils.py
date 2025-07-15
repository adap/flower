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
"""SuperLink utilities."""


from typing import Optional, Union

import grpc

from flwr.common.constant import Status, SubStatus
from flwr.common.typing import RunStatus
from flwr.proto.appio_pb2 import PushAppMessagesRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import PushMessagesRequest  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectIDs  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.object_store import ObjectStore

_STATUS_TO_MSG = {
    Status.PENDING: "Run is pending.",
    Status.STARTING: "Run is starting.",
    Status.RUNNING: "Run is running.",
    Status.FINISHED: "Run is finished.",
}


def check_abort(
    run_id: int,
    abort_status_list: list[str],
    state: LinkState,
    store: Optional[ObjectStore] = None,
) -> Union[str, None]:
    """Check if the status of the provided `run_id` is in `abort_status_list`."""
    run_status: RunStatus = state.get_run_status({run_id})[run_id]

    if run_status.status in abort_status_list:
        msg = _STATUS_TO_MSG[run_status.status]
        if run_status.sub_status == SubStatus.STOPPED:
            msg += " Stopped by user."
        return msg

    # Clear the objects of the run from the store if the run is finished
    if store and run_status.status == Status.FINISHED:
        store.delete_objects_in_run(run_id)

    return None


def abort_grpc_context(msg: Union[str, None], context: grpc.ServicerContext) -> None:
    """Abort context with statuscode PERMISSION_DENIED if `msg` is not None."""
    if msg is not None:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, msg)


def abort_if(
    run_id: int,
    abort_status_list: list[str],
    state: LinkState,
    store: Optional[ObjectStore],
    context: grpc.ServicerContext,
) -> None:
    """Abort context if status of the provided `run_id` is in `abort_status_list`."""
    msg = check_abort(run_id, abort_status_list, state, store)
    abort_grpc_context(msg, context)


def store_mapping_and_register_objects(
    store: ObjectStore, request: Union[PushAppMessagesRequest, PushMessagesRequest]
) -> dict[str, ObjectIDs]:
    """Store Message object to descendants mapping and preregister objects."""
    if not request.messages_list:
        return {}

    objects_to_push: dict[str, ObjectIDs] = {}

    # Get run_id from the first message in the list
    # All messages of a request should in the same run
    run_id = request.messages_list[0].metadata.run_id

    for object_tree in request.message_object_trees:
        # Preregister
        object_ids_just_registered = store.preregister(run_id, object_tree)
        # Keep track of objects that need to be pushed
        objects_to_push[object_tree.object_id] = ObjectIDs(
            object_ids=object_ids_just_registered
        )

    return objects_to_push

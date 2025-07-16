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
"""ServerAppIo API servicer."""


import threading
from logging import DEBUG, ERROR, INFO
from typing import Optional

import grpc

from flwr.common import Message
from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.common.inflatable import (
    UnexpectedObjectContentError,
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
    no_object_id_recompute,
)
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_status_from_proto,
    run_status_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab, RunStatus
from flwr.proto import serverappio_pb2_grpc  # pylint: disable=E0611
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
)
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    ObjectIDs,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    GetRunStatusRequest,
    GetRunStatusResponse,
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.utils import abort_if
from flwr.server.utils.validator import validate_message
from flwr.supercore.ffs import Ffs, FfsFactory
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStoreFactory

from ..utils import store_mapping_and_register_objects


class ServerAppIoServicer(serverappio_pb2_grpc.ServerAppIoServicer):
    """ServerAppIo API servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.lock = threading.RLock()

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ServerAppIoServicer.GetNodes")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        all_ids: set[int] = state.get_nodes(request.run_id)
        nodes: list[Node] = [Node(node_id=node_id) for node_id in all_ids]
        return GetNodesResponse(nodes=nodes)

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PushMessages")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Validate request and insert in State
        _raise_if(
            validation_error=len(request.messages_list) == 0,
            request_name="PushMessages",
            detail="`messages_list` must not be empty",
        )
        message_ids: list[Optional[str]] = []
        for message_proto in request.messages_list:
            message = message_from_proto(message_proto=message_proto)
            validation_errors = validate_message(message, is_reply_message=False)
            _raise_if(
                validation_error=bool(validation_errors),
                request_name="PushMessages",
                detail=", ".join(validation_errors),
            )
            _raise_if(
                validation_error=request.run_id != message.metadata.run_id,
                request_name="PushMessages",
                detail="`Message.metadata` has mismatched `run_id`",
            )
            # Store
            message_id: Optional[str] = state.store_message_ins(message=message)
            message_ids.append(message_id)

        # Store Message object to descendants mapping and preregister objects
        objects_to_push = store_mapping_and_register_objects(store, request=request)

        return PushAppMessagesResponse(
            message_ids=[
                str(message_id) if message_id else "" for message_id in message_ids
            ],
            objects_to_push=objects_to_push,
        )

    def PullMessages(  # pylint: disable=R0914
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PullMessages")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Read from state
        messages_res: list[Message] = state.get_message_res(
            message_ids=set(request.message_ids)
        )

        # Register messages generated by LinkState in the Store for consistency
        for msg_res in messages_res:
            if msg_res.metadata.src_node_id == SUPERLINK_NODE_ID:
                with no_object_id_recompute():
                    all_objects = get_all_nested_objects(msg_res)
                    # Preregister
                    store.preregister(request.run_id, get_object_tree(msg_res))
                    # Store objects
                    for obj_id, obj in all_objects.items():
                        store.put(obj_id, obj.deflate())

        # Delete the instruction Messages and their replies if found
        message_ins_ids_to_delete = {
            msg_res.metadata.reply_to_message_id for msg_res in messages_res
        }

        state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

        # Convert Messages to proto
        messages_list = []
        objects_to_pull: dict[str, ObjectIDs] = {}
        while messages_res:
            msg = messages_res.pop(0)

            # Skip `run_id` check for SuperLink generated replies
            if msg.metadata.src_node_id != SUPERLINK_NODE_ID:
                _raise_if(
                    validation_error=request.run_id != msg.metadata.run_id,
                    request_name="PullMessages",
                    detail="`message.metadata` has mismatched `run_id`",
                )
            messages_list.append(message_to_proto(msg))

            try:
                msg_object_id = msg.metadata.message_id
                obj_tree = store.get_object_tree(msg_object_id)
                descendants = [node.object_id for node in iterate_object_tree(obj_tree)]
                descendants = descendants[:-1]  # Exclude the message itself
                # Add mapping of message object ID to its descendants
                objects_to_pull[msg_object_id] = ObjectIDs(object_ids=descendants)
            except NoObjectInStoreError as e:
                log(ERROR, e.message)
                # Delete message ins from state
                state.delete_messages(message_ins_ids={msg_object_id})

        return PullAppMessagesResponse(
            messages_list=messages_list, objects_to_pull=objects_to_pull
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "ServerAppIoServicer.GetRun")

        # Init state
        state: LinkState = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(run))

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB from Ffs."""
        log(DEBUG, "ServerAppIoServicer.GetFab")

        ffs: Ffs = self.ffs_factory.ffs()
        if result := ffs.get(request.hash_str):
            fab = Fab(request.hash_str, result[0])
            return GetFabResponse(fab=fab_to_proto(fab))

        raise ValueError(f"Found no FAB with hash: {request.hash_str}")

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: grpc.ServicerContext
    ) -> PullAppInputsResponse:
        """Pull ServerApp process inputs."""
        log(DEBUG, "ServerAppIoServicer.PullAppInputs")
        # Init access to LinkState
        state = self.state_factory.state()

        # Lock access to LinkState, preventing obtaining the same pending run_id
        with self.lock:
            # Attempt getting the run_id of a pending run
            run_id = state.get_pending_run_id()
            # If there's no pending run, return an empty response
            if run_id is None:
                return PullAppInputsResponse()

            # Init access to Ffs
            ffs = self.ffs_factory.ffs()

            # Retrieve Context, Run and Fab for the run_id
            serverapp_ctxt = state.get_serverapp_context(run_id)
            run = state.get_run(run_id)
            fab = None
            if run and run.fab_hash:
                if result := ffs.get(run.fab_hash):
                    fab = Fab(run.fab_hash, result[0])
            if run and fab and serverapp_ctxt:
                # Update run status to STARTING
                if state.update_run_status(run_id, RunStatus(Status.STARTING, "", "")):
                    log(INFO, "Starting run %d", run_id)
                    return PullAppInputsResponse(
                        context=context_to_proto(serverapp_ctxt),
                        run=run_to_proto(run),
                        fab=fab_to_proto(fab),
                    )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to STARTING
        raise RuntimeError(f"Failed to start run {run_id}")

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushAppOutputsResponse:
        """Push ServerApp process outputs."""
        log(DEBUG, "ServerAppIoServicer.PushAppOutputs")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        state.set_serverapp_context(request.run_id, context_from_proto(request.context))
        return PushAppOutputsResponse()

    def UpdateRunStatus(
        self, request: UpdateRunStatusRequest, context: grpc.ServicerContext
    ) -> UpdateRunStatusResponse:
        """Update the status of a run."""
        log(DEBUG, "ServerAppIoServicer.UpdateRunStatus")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is finished
        abort_if(request.run_id, [Status.FINISHED], state, store, context)

        # Update the run status
        state.update_run_status(
            run_id=request.run_id, new_status=run_status_from_proto(request.run_status)
        )

        # If the run is finished, delete the run from ObjectStore
        if request.run_status.status == Status.FINISHED:
            # Delete all objects related to the run
            store.delete_objects_in_run(request.run_id)

        return UpdateRunStatusResponse()

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        log(DEBUG, "ServerAppIoServicer.PushLogs")
        state = self.state_factory.state()

        # Add logs to LinkState
        merged_logs = "".join(request.logs)
        state.add_serverapp_log(request.run_id, merged_logs)
        return PushLogsResponse()

    def GetRunStatus(
        self, request: GetRunStatusRequest, context: grpc.ServicerContext
    ) -> GetRunStatusResponse:
        """Get the status of a run."""
        log(DEBUG, "ServerAppIoServicer.GetRunStatus")
        state = self.state_factory.state()

        # Get run status from LinkState
        run_statuses = state.get_run_status(set(request.run_ids))
        run_status_dict = {
            run_id: run_status_to_proto(run_status)
            for run_id, run_status in run_statuses.items()
        }
        return GetRunStatusResponse(run_status_dict=run_status_dict)

    def SendAppHeartbeat(
        self, request: SendAppHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendAppHeartbeatResponse:
        """Handle a heartbeat from the ServerApp."""
        log(DEBUG, "ServerAppIoServicer.SendAppHeartbeat")

        # Init state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        # The app heartbeat can only be acknowledged if the run is in
        # starting or running status.
        success = state.acknowledge_app_heartbeat(
            run_id=request.run_id,
            heartbeat_interval=request.heartbeat_interval,
        )

        return SendAppHeartbeatResponse(success=success)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PushObject")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        if request.node.node_id != SUPERLINK_NODE_ID:
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

        # Insert in store
        stored = False
        try:
            store.put(request.object_id, request.object_content)
            stored = True
        except (NoObjectInStoreError, ValueError) as e:
            log(ERROR, str(e))
        except UnexpectedObjectContentError as e:
            # Object content is not valid
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

        return PushObjectResponse(stored=stored)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PullObject")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        if request.node.node_id != SUPERLINK_NODE_ID:
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

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

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        log(DEBUG, "ServerAppIoServicer.ConfirmMessageReceived")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")

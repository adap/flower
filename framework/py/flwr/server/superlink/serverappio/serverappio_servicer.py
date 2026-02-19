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

import grpc

from flwr.common import Message
from flwr.common.constant import SUPERLINK_NODE_ID, ExecPluginType, Status
from flwr.common.inflatable import (
    UnexpectedObjectContentError,
    get_all_nested_objects,
    get_object_tree,
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
    run_to_proto,
)
from flwr.common.typing import Fab, RunStatus
from flwr.proto import serverappio_pb2_grpc  # pylint: disable=E0611
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
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
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetRunRequest,
    GetRunResponse,
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.superexec_auth import (
    SuperExecAuthConfig,
    get_disabled_superexec_auth_config,
    superexec_auth_metadata_present,
    verify_superexec_signed_metadata,
)
from flwr.server.superlink.utils import abort_if
from flwr.server.utils.validator import validate_message
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStoreFactory


class ServerAppIoServicer(serverappio_pb2_grpc.ServerAppIoServicer):
    """ServerAppIo API servicer."""

    _METHOD_LIST_APPS_TO_LAUNCH = "/flwr.proto.ServerAppIo/ListAppsToLaunch"
    _METHOD_REQUEST_TOKEN = "/flwr.proto.ServerAppIo/RequestToken"
    _METHOD_GET_RUN = "/flwr.proto.ServerAppIo/GetRun"

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
        superexec_auth_config: SuperExecAuthConfig | None = None,
    ) -> None:
        """Initialize ServerAppIo servicer dependencies and auth settings."""
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.superexec_auth_config = (
            superexec_auth_config or get_disabled_superexec_auth_config()
        )
        self.lock = threading.RLock()

    def ListAppsToLaunch(
        self,
        request: ListAppsToLaunchRequest,
        context: grpc.ServicerContext,
    ) -> ListAppsToLaunchResponse:
        """Get run IDs with pending messages."""
        log(DEBUG, "ServerAppIoServicer.ListAppsToLaunch")
        self._verify_superexec_auth_if_enabled(
            context=context, method=self._METHOD_LIST_APPS_TO_LAUNCH
        )

        # Initialize state connection
        state = self.state_factory.state()

        # Get IDs of runs in pending status
        run_ids = state.get_run_ids(flwr_aid=None)
        pending_run_ids = []
        for run_id, status in state.get_run_status(run_ids).items():
            if status.status == Status.PENDING:
                pending_run_ids.append(run_id)

        # Return run IDs
        return ListAppsToLaunchResponse(run_ids=pending_run_ids)

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "ServerAppIoServicer.RequestToken")
        self._verify_superexec_auth_if_enabled(
            context=context, method=self._METHOD_REQUEST_TOKEN
        )

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        token = state.create_token(request.run_id)

        # Transition the run to STARTING if token creation was successful
        if token:
            state.update_run_status(
                run_id=request.run_id,
                new_status=RunStatus(Status.STARTING, "", ""),
            )

        # Return the token
        return RequestTokenResponse(token=token or "")

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ServerAppIoServicer.GetNodes")
        self._verify_token_for_run(request.token, request.run_id, context)

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
        self._verify_token_for_run(request.token, request.run_id, context)

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
        message_ids: list[str | None] = []
        objects_to_push: set[str] = set()
        for message_proto, object_tree in zip(
            request.messages_list, request.message_object_trees, strict=True
        ):
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
            # Store objects
            objects_to_push |= set(store.preregister(request.run_id, object_tree))
            # Store message
            message_id: str | None = state.store_message_ins(message=message)
            message_ids.append(message_id)

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
        self._verify_token_for_run(request.token, request.run_id, context)

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
        trees = []
        while messages_res:
            msg = messages_res.pop(0)

            # Skip `run_id` check for SuperLink generated replies
            if msg.metadata.src_node_id != SUPERLINK_NODE_ID:
                _raise_if(
                    validation_error=request.run_id != msg.metadata.run_id,
                    request_name="PullMessages",
                    detail="`message.metadata` has mismatched `run_id`",
                )

            try:
                msg_object_id = msg.metadata.message_id
                obj_tree = store.get_object_tree(msg_object_id)
                # Add message and object tree to the response
                messages_list.append(message_to_proto(msg))
                trees.append(obj_tree)
            except NoObjectInStoreError as e:
                log(ERROR, e.message)
                # Delete message ins from state
                state.delete_messages(message_ins_ids={msg_object_id})

        return PullAppMessagesResponse(
            messages_list=messages_list, message_object_trees=trees
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "ServerAppIoServicer.GetRun")
        self._verify_get_run_auth_if_enabled(
            request=request, context=context, method=self._METHOD_GET_RUN
        )

        # Init state
        state: LinkState = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(run))

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: grpc.ServicerContext
    ) -> PullAppInputsResponse:
        """Pull ServerApp process inputs."""
        log(DEBUG, "ServerAppIoServicer.PullAppInputs")
        # Init access to LinkState
        state = self.state_factory.state()

        # Validate the token
        run_id = self._verify_token(request.token, context)

        # Lock access to LinkState, preventing obtaining the same pending run_id
        with self.lock:
            # Init access to Ffs
            ffs = self.ffs_factory.ffs()

            # Retrieve Context, Run and Fab for the run_id
            serverapp_ctxt = state.get_serverapp_context(run_id)
            run = state.get_run(run_id)
            fab = None
            if run and run.fab_hash:
                if result := ffs.get(run.fab_hash):
                    fab = Fab(run.fab_hash, result[0], result[1])
            if run and fab and serverapp_ctxt:
                # Update run status to RUNNING
                if state.update_run_status(run_id, RunStatus(Status.RUNNING, "", "")):
                    log(INFO, "Starting run %d", run_id)
                    return PullAppInputsResponse(
                        context=context_to_proto(serverapp_ctxt),
                        run=run_to_proto(run),
                        fab=fab_to_proto(fab),
                    )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to RUNNING
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Failed to start run {run_id}",
        )
        raise RuntimeError("Unreachable code")  # for mypy

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushAppOutputsResponse:
        """Push ServerApp process outputs."""
        log(DEBUG, "ServerAppIoServicer.PushAppOutputs")

        # Validate the token
        run_id = self._verify_token(request.token, context)
        if request.run_id != run_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token for run ID.",
            )
            raise RuntimeError("Unreachable code")  # for mypy

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
        self._verify_token_for_run(request.token, request.run_id, context)

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
            # Invalidate app token only once the run is terminal. This keeps
            # authentication available for final log/status RPCs after PushAppOutputs.
            state.delete_token(request.run_id)
            # Delete all objects related to the run
            store.delete_objects_in_run(request.run_id)

        return UpdateRunStatusResponse()

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        log(DEBUG, "ServerAppIoServicer.PushLogs")
        self._verify_token_for_run(request.token, request.run_id, context)
        state = self.state_factory.state()

        # Add logs to LinkState
        merged_logs = "".join(request.logs)
        state.add_serverapp_log(request.run_id, merged_logs)
        return PushLogsResponse()

    def SendAppHeartbeat(
        self, request: SendAppHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendAppHeartbeatResponse:
        """Handle a heartbeat from an app process."""
        log(DEBUG, "ServerAppIoServicer.SendAppHeartbeat")

        # Init state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        success = state.acknowledge_app_heartbeat(request.token)
        return SendAppHeartbeatResponse(success=success)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PushObject")
        self._verify_token_for_run(request.token, request.run_id, context)

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
        self._verify_token_for_run(request.token, request.run_id, context)

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
        self._verify_token_for_run(request.token, request.run_id, context)

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

    def _verify_token(self, token: str, context: grpc.ServicerContext) -> int:
        """Verify the token and return the associated run ID."""
        state = self.state_factory.state()
        run_id = state.get_run_id_by_token(token)
        if run_id is None or not state.verify_token(run_id, token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")
        return run_id

    def _verify_token_for_run(
        self, token: str, run_id: int, context: grpc.ServicerContext
    ) -> None:
        """Verify token and ensure it belongs to the given run ID."""
        token_run_id = self._verify_token(token, context)
        if token_run_id != run_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token for run ID.",
            )
            raise RuntimeError("This line should never be reached.")

    def _verify_superexec_auth_if_enabled(
        self, context: grpc.ServicerContext, method: str
    ) -> None:
        """Verify SuperExec signed metadata when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            return
        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=ExecPluginType.SERVER_APP,
            cfg=self.superexec_auth_config,
        )

    def _verify_get_run_auth_if_enabled(
        self, request: GetRunRequest, context: grpc.ServicerContext, method: str
    ) -> None:
        """Authorize GetRun with one mechanism when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            # Legacy behavior by design: when SuperExec auth is disabled, GetRun
            # remains unauthenticated and tokenless requests are allowed.
            return

        token_present = bool(request.token)
        signed_metadata_present = superexec_auth_metadata_present(context)
        if token_present == signed_metadata_present:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Exactly one authentication mechanism must be provided.",
            )
            raise RuntimeError("This line should never be reached.")

        if token_present:
            self._verify_token_for_run(request.token, request.run_id, context)
            return

        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=ExecPluginType.SERVER_APP,
            cfg=self.superexec_auth_config,
        )


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")

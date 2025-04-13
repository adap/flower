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
from logging import DEBUG, INFO
from typing import Optional
from uuid import UUID

import grpc

from flwr.common import ConfigRecord, Message
from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_status_from_proto,
    run_status_to_proto,
    run_to_proto,
    user_config_from_proto,
)
from flwr.common.typing import Fab, RunStatus
from flwr.proto import serverappio_pb2_grpc  # pylint: disable=E0611
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
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
    PullResMessagesRequest,
    PullResMessagesResponse,
    PullServerAppInputsRequest,
    PullServerAppInputsResponse,
    PushInsMessagesRequest,
    PushInsMessagesResponse,
    PushServerAppOutputsRequest,
    PushServerAppOutputsResponse,
)
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.utils import abort_if
from flwr.server.utils.validator import validate_message


class ServerAppIoServicer(serverappio_pb2_grpc.ServerAppIoServicer):
    """ServerAppIo API servicer."""

    def __init__(
        self, state_factory: LinkStateFactory, ffs_factory: FfsFactory
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.lock = threading.RLock()

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ServerAppIoServicer.GetNodes")

        # Init state
        state: LinkState = self.state_factory.state()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            context,
        )

        all_ids: set[int] = state.get_nodes(request.run_id)
        nodes: list[Node] = [Node(node_id=node_id) for node_id in all_ids]
        return GetNodesResponse(nodes=nodes)

    def CreateRun(
        self, request: CreateRunRequest, context: grpc.ServicerContext
    ) -> CreateRunResponse:
        """Create run ID."""
        log(DEBUG, "ServerAppIoServicer.CreateRun")
        state: LinkState = self.state_factory.state()
        if request.HasField("fab"):
            fab = fab_from_proto(request.fab)
            ffs: Ffs = self.ffs_factory.ffs()
            fab_hash = ffs.put(fab.content, {})
            _raise_if(
                validation_error=fab_hash != fab.hash_str,
                request_name="CreateRun",
                detail=f"FAB ({fab.hash_str}) hash from request doesn't match contents",
            )
        else:
            fab_hash = ""
        run_id = state.create_run(
            request.fab_id,
            request.fab_version,
            fab_hash,
            user_config_from_proto(request.override_config),
            ConfigRecord(),
        )
        return CreateRunResponse(run_id=run_id)

    def PushMessages(
        self, request: PushInsMessagesRequest, context: grpc.ServicerContext
    ) -> PushInsMessagesResponse:
        """Push a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PushMessages")

        # Init state
        state: LinkState = self.state_factory.state()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            context,
        )

        # Validate request and insert in State
        _raise_if(
            validation_error=len(request.messages_list) == 0,
            request_name="PushMessages",
            detail="`messages_list` must not be empty",
        )
        message_ids: list[Optional[UUID]] = []
        while request.messages_list:
            message_proto = request.messages_list.pop(0)
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
            message_id: Optional[UUID] = state.store_message_ins(message=message)
            message_ids.append(message_id)

        return PushInsMessagesResponse(
            message_ids=[
                str(message_id) if message_id else "" for message_id in message_ids
            ]
        )

    def PullMessages(
        self, request: PullResMessagesRequest, context: grpc.ServicerContext
    ) -> PullResMessagesResponse:
        """Pull a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PullMessages")

        # Init state
        state: LinkState = self.state_factory.state()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            context,
        )

        # Convert each message_id str to UUID
        message_ids: set[UUID] = {
            UUID(message_id) for message_id in request.message_ids
        }

        # Read from state
        messages_res: list[Message] = state.get_message_res(message_ids=message_ids)

        # Delete the instruction Messages and their replies if found
        message_ins_ids_to_delete = {
            UUID(msg_res.metadata.reply_to_message_id) for msg_res in messages_res
        }

        state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

        # Convert Messages to proto
        messages_list = []
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

        return PullResMessagesResponse(messages_list=messages_list)

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

    def PullServerAppInputs(
        self, request: PullServerAppInputsRequest, context: grpc.ServicerContext
    ) -> PullServerAppInputsResponse:
        """Pull ServerApp process inputs."""
        log(DEBUG, "ServerAppIoServicer.PullServerAppInputs")
        # Init access to LinkState
        state = self.state_factory.state()

        # Lock access to LinkState, preventing obtaining the same pending run_id
        with self.lock:
            # Attempt getting the run_id of a pending run
            run_id = state.get_pending_run_id()
            # If there's no pending run, return an empty response
            if run_id is None:
                return PullServerAppInputsResponse()

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
                    return PullServerAppInputsResponse(
                        context=context_to_proto(serverapp_ctxt),
                        run=run_to_proto(run),
                        fab=fab_to_proto(fab),
                    )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to STARTING
        raise RuntimeError(f"Failed to start run {run_id}")

    def PushServerAppOutputs(
        self, request: PushServerAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushServerAppOutputsResponse:
        """Push ServerApp process outputs."""
        log(DEBUG, "ServerAppIoServicer.PushServerAppOutputs")

        # Init state
        state = self.state_factory.state()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            context,
        )

        state.set_serverapp_context(request.run_id, context_from_proto(request.context))
        return PushServerAppOutputsResponse()

    def UpdateRunStatus(
        self, request: UpdateRunStatusRequest, context: grpc.ServicerContext
    ) -> UpdateRunStatusResponse:
        """Update the status of a run."""
        log(DEBUG, "ServerAppIoServicer.UpdateRunStatus")

        # Init state
        state = self.state_factory.state()

        # Abort if the run is finished
        abort_if(request.run_id, [Status.FINISHED], state, context)

        # Update the run status
        state.update_run_status(
            run_id=request.run_id, new_status=run_status_from_proto(request.run_status)
        )
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


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")

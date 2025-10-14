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
"""Fleet API gRPC request-response servicer."""


from logging import DEBUG, ERROR, INFO

import grpc
from google.protobuf.json_format import MessageToDict

from flwr.common.constant import SUPERNODE_NOT_CREATED_FROM_CLI_MESSAGE
from flwr.common.inflatable import UnexpectedObjectContentError
from flwr.common.logger import log
from flwr.common.typing import InvalidRunStatusException
from flwr.proto import fleet_pb2_grpc  # pylint: disable=E0611
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
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.utils import abort_grpc_context
from flwr.supercore.constant import NodeStatus
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
        enable_supernode_auth: bool,
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.enable_supernode_auth = enable_supernode_auth

    def CreateNode(
        self, request: CreateNodeRequest, context: grpc.ServicerContext
    ) -> CreateNodeResponse:
        """."""
        log(
            INFO,
            "[Fleet.CreateNode] Request heartbeat_interval=%s",
            request.heartbeat_interval,
        )
        log(DEBUG, "[Fleet.CreateNode] Request: %s", MessageToDict(request))
        try:

            state = self.state_factory.state()

            # Check if public key is already in use
            if node_id := state.get_node_id_by_public_key(request.public_key):

                node_info = state.get_node_info(node_ids=[node_id])[0]
                if node_info.status == NodeStatus.ONLINE:
                    # Node is already active
                    log(
                        ERROR,
                        "Public key already in use (node_id=%s)",
                        node_id,
                    )
                    raise ValueError("Public key already in use by an active SuperNode")

                # Prepare response with existing node_id
                response = CreateNodeResponse(node=Node(node_id=node_id))
            else:
                if self.enable_supernode_auth:
                    # When SuperNode authentication is enabled,
                    # only SuperNodes created from the CLI are allowed to
                    # stablish a connection with the Fleet API
                    log(ERROR, SUPERNODE_NOT_CREATED_FROM_CLI_MESSAGE)
                    raise ValueError(SUPERNODE_NOT_CREATED_FROM_CLI_MESSAGE)

                # When SuperNode authentication is disabled, auto-auth
                # allows creating a new node
                response = message_handler.create_node(
                    request=request,
                    state=state,
                )

        except ValueError as e:
            # Public key already in use
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        log(INFO, "[Fleet.CreateNode] Created node_id=%s", response.node.node_id)
        log(DEBUG, "[Fleet.CreateNode] Response: %s", MessageToDict(response))
        return response

    def DeleteNode(
        self, request: DeleteNodeRequest, context: grpc.ServicerContext
    ) -> DeleteNodeResponse:
        """."""
        log(INFO, "[Fleet.DeleteNode] Delete node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.DeleteNode] Request: %s", MessageToDict(request))
        # This shall be refactored when renaming `Fleet.Create/DeleteNode`
        # to `Fleet.Activate/DeactivateNode`
        if self.enable_supernode_auth:
            # SuperNodes can only be deleted from the CLI
            # We simply acknowledge the heartbeat with interval 0
            # to mark the node as offline
            state = self.state_factory.state()
            state.acknowledge_node_heartbeat(
                node_id=request.node.node_id, heartbeat_interval=0
            )
            return DeleteNodeResponse()

        return message_handler.delete_node(
            request=request,
            state=self.state_factory.state(),
        )

    def SendNodeHeartbeat(
        self, request: SendNodeHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendNodeHeartbeatResponse:
        """."""
        log(DEBUG, "[Fleet.SendNodeHeartbeat] Request: %s", MessageToDict(request))
        return message_handler.send_node_heartbeat(
            request=request,
            state=self.state_factory.state(),
        )

    def PullMessages(
        self, request: PullMessagesRequest, context: grpc.ServicerContext
    ) -> PullMessagesResponse:
        """Pull Messages."""
        log(INFO, "[Fleet.PullMessages] node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.PullMessages] Request: %s", MessageToDict(request))
        return message_handler.pull_messages(
            request=request,
            state=self.state_factory.state(),
            store=self.objectstore_factory.store(),
        )

    def PushMessages(
        self, request: PushMessagesRequest, context: grpc.ServicerContext
    ) -> PushMessagesResponse:
        """Push Messages."""
        if request.messages_list:
            log(
                INFO,
                "[Fleet.PushMessages] Push replies from node_id=%s",
                request.messages_list[0].metadata.src_node_id,
            )
        else:
            log(INFO, "[Fleet.PushMessages] No replies to push")

        try:
            res = message_handler.push_messages(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)

        return res

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(INFO, "[Fleet.GetRun] Requesting `Run` for run_id=%s", request.run_id)

        try:
            res = message_handler.get_run(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)

        return res

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB."""
        log(INFO, "[Fleet.GetFab] Requesting FAB for fab_hash=%s", request.hash_str)
        try:
            res = message_handler.get_fab(
                request=request,
                ffs=self.ffs_factory.ffs(),
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)

        return res

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(
            DEBUG,
            "[ServerAppIoServicer.PushObject] Push Object with object_id=%s",
            request.object_id,
        )

        try:
            # Insert in Store
            res = message_handler.push_object(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)
        except UnexpectedObjectContentError as e:
            # Object content is not valid
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

        return res

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(
            DEBUG,
            "[ServerAppIoServicer.PullObject] Pull Object with object_id=%s",
            request.object_id,
        )

        try:
            # Fetch from store
            res = message_handler.pull_object(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)

        return res

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        log(
            DEBUG,
            "[Fleet.ConfirmMessageReceived] Message with ID '%s' has been received",
            request.message_object_id,
        )

        try:
            res = message_handler.confirm_message_received(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            abort_grpc_context(e.message, context)

        return res

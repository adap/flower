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

from flwr.common.constant import Status
from flwr.common.inflatable import check_body_len_consistency
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
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.utils import abort_grpc_context, check_abort
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.object_store.object_store import NoObjectInStoreError


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory

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
        response = message_handler.create_node(
            request=request,
            state=self.state_factory.state(),
        )
        log(INFO, "[Fleet.CreateNode] Created node_id=%s", response.node.node_id)
        log(DEBUG, "[Fleet.CreateNode] Response: %s", MessageToDict(response))
        return response

    def DeleteNode(
        self, request: DeleteNodeRequest, context: grpc.ServicerContext
    ) -> DeleteNodeResponse:
        """."""
        log(INFO, "[Fleet.DeleteNode] Delete node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.DeleteNode] Request: %s", MessageToDict(request))
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

        state = self.state_factory.state()

        # Abort if the run is not running
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            abort_grpc_context(abort_msg, context)

        if request.node.node_id not in state.get_nodes(run_id=request.run_id):
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

        if not check_body_len_consistency(request.object_content):
            # Cancel insertion in ObjectStore
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Unexpected object length"
            )

        # Init store
        store = self.objectstore_factory.store()

        # Insert in store
        stored = False
        try:
            store.put(request.object_id, request.object_content)
            stored = True
        except (NoObjectInStoreError, ValueError) as e:
            log(ERROR, str(e))

        return PushObjectResponse(stored=stored)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(
            DEBUG,
            "[ServerAppIoServicer.PullObject] Pull Object with object_id=%s",
            request.object_id,
        )

        state = self.state_factory.state()

        # Abort if the run is not running
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            abort_grpc_context(abort_msg, context)

        if request.node.node_id not in state.get_nodes(run_id=request.run_id):
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

        # Init store
        store = self.objectstore_factory.store()

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

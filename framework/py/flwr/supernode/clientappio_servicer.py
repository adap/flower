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
"""ClientAppIo API servicer."""


import secrets
import threading
from logging import DEBUG

import grpc

from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab

# pylint: disable=E0611
from flwr.proto import clientappio_pb2_grpc
from flwr.proto.clientappio_pb2 import (
    GetRunIdsWithPendingMessagesRequest,
    GetRunIdsWithPendingMessagesResponse,
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse

# pylint: enable=E0611
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory


# pylint: disable=C0103,W0613,W0201
class ClientAppIoServicer(clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(
        self,
        state_factory: NodeStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.lock = threading.Lock()

    def GetRunIdsWithPendingMessages(
        self,
        request: GetRunIdsWithPendingMessagesRequest,
        context: grpc.ServicerContext,
    ) -> GetRunIdsWithPendingMessagesResponse:
        """Get run IDs with pending messages."""
        log(DEBUG, "ClientAppIo.GetRunIdsWithPendingMessages")

        # Initialize state connection
        state = self.state_factory.state()

        # Get run IDs with pending messages
        run_ids = state.get_run_ids_with_pending_messages()

        # Return run IDs
        return GetRunIdsWithPendingMessagesResponse(run_ids=run_ids)

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get Run by run ID."""
        log(DEBUG, "ClientAppIo.GetRun")

        # Initialize state connection
        state = self.state_factory.state()

        # Get Run by run ID
        run = state.get_run(run_id=request.run_id)

        # Return Run
        if run is None:
            return GetRunResponse()
        return GetRunResponse(run=run_to_proto(run))

    def PullClientAppInputs(
        self, request: PullClientAppInputsRequest, context: grpc.ServicerContext
    ) -> PullClientAppInputsResponse:
        """Pull Message, Context, Run, and Fab."""
        log(DEBUG, "ClientAppIo.PullClientAppInputs")

        # Initialize state connection
        state = self.state_factory.state()
        ffs = self.ffs_factory.ffs()

        # TODO: Validate request in a interceptor
        if not state.verify_token(request.run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid or missing token for the requested run ID.",
            )

        # Get inputs from state
        found_messages = state.get_message(
            run_id=request.run_id, is_reply=False, limit=1
        )
        if not found_messages:
            return PullClientAppInputsResponse()
        message = next(iter(found_messages.values()))
        run_ctx = state.get_context(run_id=request.run_id)
        run = state.get_run(run_id=request.run_id)
        fab = None
        if result := ffs.get(run.fab_hash):
            fab = Fab(run.fab_hash, result[0])

        # Return inputs
        return PullClientAppInputsResponse(
            message=message_to_proto(message),
            context=context_to_proto(run_ctx) if run_ctx else None,
            run=run_to_proto(run) if run else None,
            fab=fab_to_proto(fab) if fab else None,
        )

    def PushClientAppOutputs(
        self, request: PushClientAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushClientAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushClientAppOutputs")

        # Initialize state connection
        state = self.state_factory.state()

        # TODO: Validate request in a interceptor
        if not state.verify_token(request.run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid or missing token for the requested run ID.",
            )

        # Get Message and Context from request
        message = message_from_proto(request.message)
        run_ctx = context_from_proto(request.context)

        # Verify run ID
        if (
            run_ctx.run_id != request.run_id
            or message.metadata.run_id != request.run_id
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Run ID in the context or message does not match the requested run ID.",
            )

        # Store Message and Context in state
        # TODO: use real object ID instead of a random one
        object_id = secrets.token_hex(64)
        state.store_message(message, object_id)
        state.store_context(run_ctx)

        # Delete the token for the run ID
        state.delete_token(request.run_id)

        # Return empty response
        return PushClientAppOutputsResponse()

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "ClientAppIo.RequestToken")

        # Initialize state connection
        state = self.state_factory.state()

        # Create a token for the given run ID
        try:
            token = state.create_token(request.run_id)
        except ValueError:
            # If a token already exists for this run ID (the run is in progress),
            # return an empty token
            return RequestTokenResponse(token=b"")

        # Return the token
        return RequestTokenResponse(token=token)

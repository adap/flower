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


from logging import DEBUG
from typing import cast

import grpc

from flwr.common import Context
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab, Run

# pylint: disable=E0611
from flwr.proto import clientappio_pb2_grpc
from flwr.proto.clientappio_pb2 import (  # pylint: disable=E0401
    GetRunIdsWithPendingMessagesRequest,
    GetRunIdsWithPendingMessagesResponse,
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.supercore.ffs import FfsFactory
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

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "ClientAppIo.RequestToken")

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        try:
            token = state.create_token(request.run_id)
        except ValueError:
            # Return an empty token if A token already exists for this run ID,
            # indicating the run is in progress
            return RequestTokenResponse(token="")

        # Return the token
        return RequestTokenResponse(token=token)

    def PullClientAppInputs(
        self, request: PullClientAppInputsRequest, context: grpc.ServicerContext
    ) -> PullClientAppInputsResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullClientAppInputs")

        # Initialize state and ffs connection
        state = self.state_factory.state()
        ffs = self.ffs_factory.ffs()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Retrieve message, context, run and fab for this run
        message = state.get_messages(run_ids=[run_id], is_reply=False)[0]
        context = cast(Context, state.get_context(run_id))
        run = cast(Run, state.get_run(run_id))
        fab = Fab(run.fab_hash, ffs.get(run.fab_hash)[0])  # type: ignore

        return PullClientAppInputsResponse(
            message=message_to_proto(message),
            context=context_to_proto(context),
            run=run_to_proto(run),
            fab=fab_to_proto(fab),
        )

    def PushClientAppOutputs(
        self, request: PushClientAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushClientAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushClientAppOutputs")

        # Initialize state connection
        state = self.state_factory.state()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Save the message and context to the state
        state.store_message(message_from_proto(request.message))
        state.store_context(context_from_proto(request.context))

        # Remove the token to make the run eligible for processing
        # A run associated with a token cannot be handled until its token is cleared
        state.delete_token(run_id)

        return PushClientAppOutputsResponse()

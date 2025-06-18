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


from dataclasses import dataclass
from logging import DEBUG, ERROR
from typing import Optional, cast

import grpc

from flwr.common import Context, Message, typing
from flwr.common.logger import log
from flwr.common.serde import (
    clientappstatus_to_proto,
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
    GetTokenRequest,
    GetTokenResponse,
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


@dataclass
class ClientAppInputs:
    """Specify the inputs to the ClientApp."""

    message: Message
    context: Context
    run: Run
    fab: Optional[Fab]


@dataclass
class ClientAppOutputs:
    """Specify the outputs from the ClientApp."""

    message: Message
    context: Context


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

        self.clientapp_input: Optional[ClientAppInputs] = None
        self.clientapp_output: Optional[ClientAppOutputs] = None
        self.token_returned: bool = False

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

    def GetToken(
        self, request: GetTokenRequest, context: grpc.ServicerContext
    ) -> GetTokenResponse:
        """Get token."""
        log(DEBUG, "ClientAppIo.GetToken")

        # Fail if no ClientAppInputs are available
        if self.clientapp_input is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "No inputs available.",
            )

        # Fail if token was already returned in a previous call
        if self.token_returned:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Token already returned. A token can be returned only once.",
            )

        # If
        # - ClientAppInputs is set, and
        # - token hasn't been returned before,
        # return token
        self.token_returned = True
        return GetTokenResponse(token=123)  # To be deleted

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

        # Delete the token
        state.delete_token(run_id)

        # Preconditions met
        try:
            # Update Message and Context
            self.clientapp_output = ClientAppOutputs(
                message=message_from_proto(request.message),
                context=context_from_proto(request.context),
            )

            # Set status
            code = typing.ClientAppOutputCode.SUCCESS
            status = typing.ClientAppOutputStatus(code=code, message="Success")
        except Exception as e:  # pylint: disable=broad-exception-caught
            log(ERROR, "ClientApp failed to push message to SuperNode, %s", e)
            code = typing.ClientAppOutputCode.UNKNOWN_ERROR
            status = typing.ClientAppOutputStatus(code=code, message="Unkonwn error")

        # Return status to ClientApp process
        proto_status = clientappstatus_to_proto(status=status)
        return PushClientAppOutputsResponse(status=proto_status)

    def has_outputs(self) -> bool:
        """Check if ClientAppOutputs are available."""
        return self.clientapp_output is not None

    def get_outputs(self) -> ClientAppOutputs:
        """Get ClientApp outputs."""
        if self.clientapp_output is None:
            raise ValueError("ClientAppOutputs not set before calling `get_outputs`.")

        # Set outputs to a local variable and clear state
        output: ClientAppOutputs = self.clientapp_output
        self.clientapp_output = None

        return output

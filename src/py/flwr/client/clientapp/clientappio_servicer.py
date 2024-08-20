# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
    GetTokenRequest,
    GetTokenResponse,
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
)


@dataclass
class ClientAppInputs:
    """Specify the inputs to the ClientApp."""

    message: Message
    context: Context
    run: Run
    fab: Optional[Fab]
    token: int


@dataclass
class ClientAppOutputs:
    """Specify the outputs from the ClientApp."""

    message: Message
    context: Context


# pylint: disable=C0103,W0613,W0201
class ClientAppIoServicer(clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(self) -> None:
        self.clientapp_input: Optional[ClientAppInputs] = None
        self.clientapp_output: Optional[ClientAppOutputs] = None
        self.token_returned: bool = False
        self.inputs_returned: bool = False

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
        clientapp_input = cast(ClientAppInputs, self.clientapp_input)

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
        return GetTokenResponse(token=clientapp_input.token)

    def PullClientAppInputs(
        self, request: PullClientAppInputsRequest, context: grpc.ServicerContext
    ) -> PullClientAppInputsResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullClientAppInputs")

        # Fail if no ClientAppInputs are available
        if self.clientapp_input is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "No inputs available.",
            )
        clientapp_input = cast(ClientAppInputs, self.clientapp_input)

        # Fail if token wasn't returned in a previous call
        if not self.token_returned:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Token hasn't been returned."
                "Token must be returned before can be returned only once.",
            )

        # Fail if token isn't matching
        if request.token != clientapp_input.token:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Mismatch between ClientApp and SuperNode token",
            )

        # Success
        self.inputs_returned = True
        return PullClientAppInputsResponse(
            message=message_to_proto(clientapp_input.message),
            context=context_to_proto(clientapp_input.context),
            run=run_to_proto(clientapp_input.run),
            fab=fab_to_proto(clientapp_input.fab) if clientapp_input.fab else None,
        )

    def PushClientAppOutputs(
        self, request: PushClientAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushClientAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushClientAppOutputs")

        # Fail if no ClientAppInputs are available
        if not self.clientapp_input:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "No inputs available.",
            )
        clientapp_input = cast(ClientAppInputs, self.clientapp_input)

        # Fail if token wasn't returned in a previous call
        if not self.token_returned:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Token hasn't been returned."
                "Token must be returned before can be returned only once.",
            )

        # Fail if inputs weren't delivered in a previous call
        if not self.inputs_returned:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Inputs haven't been delivered."
                "Inputs must be delivered before can be returned only once.",
            )

        # Fail if token isn't matching
        if request.token != clientapp_input.token:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Mismatch between ClientApp and SuperNode token",
            )

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

    def set_inputs(
        self, clientapp_input: ClientAppInputs, token_returned: bool
    ) -> None:
        """Set ClientApp inputs.

        Parameters
        ----------
        clientapp_input : ClientAppInputs
            The inputs to the ClientApp.
        token_returned : bool
            A boolean indicating if the token has been returned.
            Set to `True` when passing the token to `flwr-clientap`
            and `False` otherwise.
        """
        if (
            self.clientapp_input is not None
            or self.clientapp_output is not None
            or self.token_returned
        ):
            raise ValueError(
                "ClientAppInputs and ClientAppOutputs must not be set before "
                "calling `set_inputs`."
            )
        log(DEBUG, "ClientAppInputs set (token: %s)", clientapp_input.token)
        self.clientapp_input = clientapp_input
        self.token_returned = token_returned

    def has_outputs(self) -> bool:
        """Check if ClientAppOutputs are available."""
        return self.clientapp_output is not None

    def get_outputs(self) -> ClientAppOutputs:
        """Get ClientApp outputs."""
        if self.clientapp_output is None:
            raise ValueError("ClientAppOutputs not set before calling `get_outputs`.")

        # Set outputs to a local variable and clear state
        output: ClientAppOutputs = self.clientapp_output
        self.clientapp_input = None
        self.clientapp_output = None
        self.token_returned = False
        self.inputs_returned = False

        return output

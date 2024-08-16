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
from typing import Optional

import grpc

from flwr.common import Context, Message, typing
from flwr.common.logger import log
from flwr.common.serde import (
    clientappstatus_to_proto,
    context_from_proto,
    context_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.common.typing import Run

# pylint: disable=E0611
from flwr.proto import clientappio_pb2_grpc
from flwr.proto.clientappio_pb2 import (  # pylint: disable=E0401
    PullClientAppInputsRequest,
    PullClientAppInputsResponse,
    PushClientAppOutputsRequest,
    PushClientAppOutputsResponse,
)


@dataclass
class ClientAppIoInputs:
    """Specify the inputs to the ClientApp."""

    message: Message
    context: Context
    run: Run
    token: int


@dataclass
class ClientAppIoOutputs:
    """Specify the outputs from the ClientApp."""

    message: Message
    context: Context


# pylint: disable=C0103,W0613,W0201
class ClientAppIoServicer(clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(self) -> None:
        self.clientapp_input: Optional[ClientAppIoInputs] = None
        self.clientapp_output: Optional[ClientAppIoOutputs] = None

    def PullClientAppInputs(
        self, request: PullClientAppInputsRequest, context: grpc.ServicerContext
    ) -> PullClientAppInputsResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullClientAppInputs")
        if self.clientapp_input is None:
            raise ValueError(
                "ClientAppIoInputs not set before calling `PullClientAppInputs`."
            )
        if request.token != self.clientapp_input.token:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Mismatch between ClientApp and SuperNode token",
            )
        return PullClientAppInputsResponse(
            message=message_to_proto(self.clientapp_input.message),
            context=context_to_proto(self.clientapp_input.context),
            run=run_to_proto(self.clientapp_input.run),
        )

    def PushClientAppOutputs(
        self, request: PushClientAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushClientAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushClientAppOutputs")
        if self.clientapp_output is None:
            raise ValueError(
                "ClientAppIoOutputs not set before calling `PushClientAppOutputs`."
            )
        if self.clientapp_input is None:
            raise ValueError(
                "ClientAppIoInputs not set before calling `PushClientAppOutputs`."
            )
        if request.token != self.clientapp_input.token:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Mismatch between ClientApp and SuperNode token",
            )
        try:
            # Update Message and Context
            self.clientapp_output.message = message_from_proto(request.message)
            self.clientapp_output.context = context_from_proto(request.context)
            # Set status
            code = typing.ClientAppOutputCode.SUCCESS
            status = typing.ClientAppOutputStatus(code=code, message="Success")
            proto_status = clientappstatus_to_proto(status=status)
            return PushClientAppOutputsResponse(status=proto_status)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log(ERROR, "ClientApp failed to push message to SuperNode, %s", e)
            code = typing.ClientAppOutputCode.UNKNOWN_ERROR
            status = typing.ClientAppOutputStatus(code=code, message="Push failed")
            proto_status = clientappstatus_to_proto(status=status)
            return PushClientAppOutputsResponse(status=proto_status)

    def set_inputs(self, clientapp_input: ClientAppIoInputs) -> None:
        """Set ClientApp inputs."""
        log(DEBUG, "ClientAppIo.SetInputs")
        if self.clientapp_input is not None or self.clientapp_output is not None:
            raise ValueError(
                "ClientAppIoInputs and ClientAppIoOutputs must not be set before "
                "calling `set_inputs`."
            )
        self.clientapp_input = clientapp_input

    def get_outputs(self) -> ClientAppIoOutputs:
        """Get ClientApp outputs."""
        log(DEBUG, "ClientAppIo.GetOutputs")
        if self.clientapp_output is None:
            raise ValueError("ClientAppIoOutputs not set before calling `get_outputs`.")
        # Set outputs to a local variable and clear self.clientapp_output
        output: ClientAppIoOutputs = self.clientapp_output
        self.clientapp_input = None
        self.clientapp_output = None
        return output

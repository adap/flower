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
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.run_pb2 import Run as ProtoRun


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
        self.message: Optional[Message] = None
        self.context: Optional[Context] = None
        self.proto_message: Optional[ProtoMessage] = None
        self.proto_context: Optional[ProtoContext] = None
        self.proto_run: Optional[ProtoRun] = None
        self.token: int = 0

    def PullClientAppInputs(
        self, request: PullClientAppInputsRequest, context: grpc.ServicerContext
    ) -> PullClientAppInputsResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullInputs")
        if request.token != self.token:
            raise ValueError("Mismatch between ClientApp and SuperNode token")
        return PullClientAppInputsResponse(
            message=self.proto_message,
            context=self.proto_context,
            run=self.proto_run,
        )

    def PushClientAppOutputs(
        self, request: PushClientAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushClientAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushOutputs")
        if request.token != self.token:
            raise ValueError("Mismatch between ClientApp and SuperNode token")
        self.proto_message = request.message
        self.proto_context = request.context
        # Update Message and Context
        try:
            self._update_payload()
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

    def set_inputs(
        self,
        clientapp_input: ClientAppIoInputs,
    ) -> None:
        """Set ClientApp inputs."""
        log(DEBUG, "ClientAppIo.SetObject")
        # Serialize Message, Context, and Run
        self.proto_message = message_to_proto(clientapp_input.message)
        self.proto_context = context_to_proto(clientapp_input.context)
        self.proto_run = run_to_proto(clientapp_input.run)
        self.token = clientapp_input.token

    def get_outputs(self) -> ClientAppIoOutputs:
        """Get ClientApp outputs."""
        log(DEBUG, "ClientAppIo.GetObject")
        if self.message is None or self.context is None:
            raise ValueError(
                "Both message and context must be set before calling `get_payload`."
            )
        clientapp_output = ClientAppIoOutputs(
            message=self.message,
            context=self.context,
        )

        return clientapp_output

    def _update_payload(self) -> None:
        """Update ClientApp objects."""
        log(DEBUG, "ClientAppIo.UpdateObject")
        # Deserialize Message and Context
        if self.proto_message is not None:
            self.message = message_from_proto(self.proto_message)
        if self.proto_context is not None:
            self.context = context_from_proto(self.proto_context)

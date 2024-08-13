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


from logging import DEBUG

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


# pylint: disable=C0103,W0613
class ClientAppIoServicer(clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(self) -> None:
        self.message: Message = None
        self.context: Context = None
        self.proto_message: ProtoMessage = None
        self.proto_context: ProtoContext = None
        self.proto_run: ProtoRun = None
        self.token: int = None

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
        self._update_object()
        # Set status
        code = typing.ClientAppOutputCode.SUCCESS
        status = typing.ClientAppOutputStatus(code=code, message="Success")
        proto_status = clientappstatus_to_proto(status=status)
        return PushClientAppOutputsResponse(status=proto_status)

    def set_object(
        self,
        message: Message,
        context: Context,
        run: Run,
        token: int,
    ) -> None:
        """Set client app objects."""
        log(DEBUG, "ClientAppIo.SetObject")
        # Serialize Message, Context, and Run
        self.proto_message = message_to_proto(message)
        self.proto_context = context_to_proto(context)
        self.proto_run = run_to_proto(run)
        self.token = token

    def get_object(self) -> tuple[Message | None, Context | None]:
        """Get client app objects."""
        log(DEBUG, "ClientAppIo.GetObject")
        return self.message, self.context

    def _update_object(self) -> None:
        """Update client app objects."""
        log(DEBUG, "ClientAppIo.UpdateObject")
        # Deserialize Message and Context
        self.message = message_from_proto(self.proto_message)
        self.context = context_from_proto(self.proto_context)

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
"""Grpc Adapter."""


from typing import Any, Type, TypeVar, cast

from google.protobuf.message import Message as GrpcMessage

import flwr
from flwr.proto.fleet_pb2 import (
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    GetRunRequest,
    GetRunResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.grpcadapter_pb2 import MessageContainer
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub

KEY_FLOWER_VERSION = "flower-version"
T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapter:
    """Grpc Adapter."""

    def __init__(self, channel: Any) -> None:
        self.stub = GrpcAdapterStub(channel)

    def _send_and_receive(self, request: GrpcMessage, response_type: Type[T]) -> T:
        container_req = MessageContainer(
            metadata={KEY_FLOWER_VERSION: flwr.__version__},
            grpc_message_name=request.__class__.__qualname__,
            grpc_message_content=request.SerializeToString(),
        )
        container_res = cast(MessageContainer, self.stub.SendReceive(container_req))
        if container_res.grpc_message_name != response_type.__qualname__:
            raise ValueError(
                f"Invalid grpc_message_name. Expected {response_type.__qualname__}"
                f", but got {container_res.grpc_message_name}."
            )
        response = response_type()
        response.ParseFromString(container_res.grpc_message_content)
        return response

    def CreateNode(self, request: CreateNodeRequest) -> CreateNodeResponse:
        """."""
        return self._send_and_receive(request, CreateNodeResponse)

    def DeleteNode(self, request: DeleteNodeRequest) -> DeleteNodeResponse:
        """."""
        return self._send_and_receive(request, DeleteNodeResponse)

    def Ping(self, request: PingRequest) -> PingResponse:
        """."""
        return self._send_and_receive(request, PingResponse)

    def PullTaskIns(self, request: PullTaskInsRequest) -> PullTaskInsResponse:
        """."""
        return self._send_and_receive(request, PullTaskInsResponse)

    def PushTaskRes(self, request: PushTaskResRequest) -> PushTaskResResponse:
        """."""
        return self._send_and_receive(request, PushTaskResResponse)

    def GetRun(self, request: GetRunRequest) -> GetRunResponse:
        """."""
        return self._send_and_receive(request, GetRunResponse)

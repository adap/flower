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

from flwr.common.constant import GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY
from flwr.common.version import package_version
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
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
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub

T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapter:
    """The Adapter class to send and receive gRPC messages via the GrpcAdapterStub.

    This class utilizes the GrpcAdapterStub to send and receive gRPC messages which are
    defined and used by the Fleet API, as defined in `fleet.proto`.
    """

    def __init__(self, channel: Any) -> None:
        self.stub = GrpcAdapterStub(channel)

    def _send_and_receive(
        self, request: GrpcMessage, response_type: Type[T], **kwargs: Any
    ) -> T:
        container_req = MessageContainer(
            metadata={GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY: package_version},
            grpc_message_name=request.__class__.__qualname__,
            grpc_message_content=request.SerializeToString(),
        )
        container_res = cast(
            MessageContainer, self.stub.SendReceive(container_req, **kwargs)
        )
        if container_res.grpc_message_name != response_type.__qualname__:
            raise ValueError(
                f"Invalid grpc_message_name. Expected {response_type.__qualname__}"
                f", but got {container_res.grpc_message_name}."
            )
        response = response_type()
        response.ParseFromString(container_res.grpc_message_content)
        return response

    # pylint: disable=C0103
    def CreateNode(
        self, request: CreateNodeRequest, **kwargs: Any
    ) -> CreateNodeResponse:
        """."""
        return self._send_and_receive(request, CreateNodeResponse, **kwargs)

    def DeleteNode(
        self, request: DeleteNodeRequest, **kwargs: Any
    ) -> DeleteNodeResponse:
        """."""
        return self._send_and_receive(request, DeleteNodeResponse, **kwargs)

    def Ping(self, request: PingRequest, **kwargs: Any) -> PingResponse:
        """."""
        return self._send_and_receive(request, PingResponse, **kwargs)

    def PullTaskIns(
        self, request: PullTaskInsRequest, **kwargs: Any
    ) -> PullTaskInsResponse:
        """."""
        return self._send_and_receive(request, PullTaskInsResponse, **kwargs)

    def PushTaskRes(
        self, request: PushTaskResRequest, **kwargs: Any
    ) -> PushTaskResResponse:
        """."""
        return self._send_and_receive(request, PushTaskResResponse, **kwargs)

    def GetRun(self, request: GetRunRequest, **kwargs: Any) -> GetRunResponse:
        """."""
        return self._send_and_receive(request, GetRunResponse, **kwargs)

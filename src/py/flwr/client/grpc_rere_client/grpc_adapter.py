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


from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub
from flwr.proto.grpcadapter_pb2 import MessageContainer
from typing import Any, TypeVar, Type, cast
from google.protobuf.message import Message as GrpcMessage
from flwr.proto.fleet_pb2 import CreateNodeRequest, CreateNodeResponse, DeleteNodeRequest, DeleteNodeResponse, PingRequest, PingResponse, PullTaskInsResponse, PullTaskInsRequest, PushTaskResRequest, PushTaskResResponse, GetRunRequest, GetRunResponse
import flwr


KEY_FLOWER_VERSION = "flower-version"
T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapter:
    def __init__(self, channel: Any) -> None:
        self.stub = GrpcAdapterStub(channel)

    def _send_and_receive(self, request: GrpcMessage, response_type: Type[T]) -> T:
        container_req = MessageContainer(
            metadata={KEY_FLOWER_VERSION: flwr.__version__},
            grpc_message_name=request.__class__.__qualname__,
            grpc_message_content=request.SerializeToString()
        )
        container_res = cast(MessageContainer, self.stub.SendReceive(container_req))
        if container_res.grpc_message_name != response_type.__qualname__:
            raise ValueError(
                f"Invalid grpc_message_name. Expected {response_type.__qualname__}"
                f", but got {container_res.grpc_message_name}."
            )
        response = response_type()
        response.ParseFromString(container_res)
        return response
    
    def CreateNode():
        ...



# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Fleet API gRPC adapter servicer."""


from logging import DEBUG, INFO
from typing import Callable, Type, TypeVar

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.logger import log
from flwr.proto import grpcadapter_pb2_grpc  # pylint: disable=E0611
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
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.state import StateFactory

T = TypeVar("T", bound=GrpcMessage)


def _handle(
    msg_container: MessageContainer,
    request_type: Type[T],
    handler: Callable[[T], GrpcMessage],
) -> MessageContainer:
    req = request_type.FromString(msg_container.grpc_message_content)
    res = handler(req)
    return MessageContainer(
        metadata={},
        grpc_message_name=res.__class__.__qualname__,
        grpc_message_content=res.SerializeToString(),
    )


class GrpcAdapterServicer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    """Fleet API via GrpcAdapter servicer."""

    def __init__(self, state_factory: StateFactory) -> None:
        self.state_factory = state_factory

    def SendReceive(
        self, request: MessageContainer, context: grpc.ServicerContext
    ) -> MessageContainer:
        """."""
        log(DEBUG, "GrpcAdapterServicer.SendReceive")
        if request.grpc_message_name == CreateNodeRequest.__qualname__:
            return _handle(request, CreateNodeRequest, self._create_node)
        if request.grpc_message_name == DeleteNodeRequest.__qualname__:
            return _handle(request, DeleteNodeRequest, self._delete_node)
        if request.grpc_message_name == PingRequest.__qualname__:
            return _handle(request, PingRequest, self._ping)
        if request.grpc_message_name == PullTaskInsRequest.__qualname__:
            return _handle(request, PullTaskInsRequest, self._pull_task_ins)
        if request.grpc_message_name == PushTaskResRequest.__qualname__:
            return _handle(request, PushTaskResRequest, self._push_task_res)
        if request.grpc_message_name == GetRunRequest.__qualname__:
            return _handle(request, GetRunRequest, self._get_run)
        raise ValueError(f"Invalid grpc_message_name: {request.grpc_message_name}")

    def _create_node(self, request: CreateNodeRequest) -> CreateNodeResponse:
        """."""
        log(INFO, "GrpcAdapter.CreateNode")
        return message_handler.create_node(
            request=request,
            state=self.state_factory.state(),
        )

    def _delete_node(self, request: DeleteNodeRequest) -> DeleteNodeResponse:
        """."""
        log(INFO, "GrpcAdapter.DeleteNode")
        return message_handler.delete_node(
            request=request,
            state=self.state_factory.state(),
        )

    def _ping(self, request: PingRequest) -> PingResponse:
        """."""
        log(DEBUG, "GrpcAdapter.Ping")
        return message_handler.ping(
            request=request,
            state=self.state_factory.state(),
        )

    def _pull_task_ins(self, request: PullTaskInsRequest) -> PullTaskInsResponse:
        """Pull TaskIns."""
        log(INFO, "GrpcAdapter.PullTaskIns")
        return message_handler.pull_task_ins(
            request=request,
            state=self.state_factory.state(),
        )

    def _push_task_res(self, request: PushTaskResRequest) -> PushTaskResResponse:
        """Push TaskRes."""
        log(INFO, "GrpcAdapter.PushTaskRes")
        return message_handler.push_task_res(
            request=request,
            state=self.state_factory.state(),
        )

    def _get_run(self, request: GetRunRequest) -> GetRunResponse:
        """Get run information."""
        log(INFO, "GrpcAdapter.GetRun")
        return message_handler.get_run(
            request=request,
            state=self.state_factory.state(),
        )

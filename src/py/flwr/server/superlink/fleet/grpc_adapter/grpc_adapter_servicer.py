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
"""Fleet API gRPC adapter servicer."""


from logging import DEBUG
from typing import Callable, TypeVar

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import (
    GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_NAME_KEY,
    GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_VERSION_KEY,
    GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY,
    GRPC_ADAPTER_METADATA_MESSAGE_MODULE_KEY,
    GRPC_ADAPTER_METADATA_MESSAGE_QUALNAME_KEY,
)
from flwr.common.logger import log
from flwr.common.version import package_name, package_version
from flwr.proto import grpcadapter_pb2_grpc  # pylint: disable=E0611
from flwr.proto.fab_pb2 import GetFabRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PingRequest,
    PullMessagesRequest,
    PushMessagesRequest,
)
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611

from ..grpc_rere.fleet_servicer import FleetServicer

T = TypeVar("T", bound=GrpcMessage)


def _handle(
    msg_container: MessageContainer,
    context: grpc.ServicerContext,
    request_type: type[T],
    handler: Callable[[T, grpc.ServicerContext], GrpcMessage],
) -> MessageContainer:
    req = request_type.FromString(msg_container.grpc_message_content)
    res = handler(req, context)
    res_cls = res.__class__
    return MessageContainer(
        metadata={
            GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_NAME_KEY: package_name,
            GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_VERSION_KEY: package_version,
            GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY: package_version,
            GRPC_ADAPTER_METADATA_MESSAGE_MODULE_KEY: res_cls.__module__,
            GRPC_ADAPTER_METADATA_MESSAGE_QUALNAME_KEY: res_cls.__qualname__,
        },
        grpc_message_name=res_cls.__qualname__,
        grpc_message_content=res.SerializeToString(),
    )


class GrpcAdapterServicer(grpcadapter_pb2_grpc.GrpcAdapterServicer, FleetServicer):
    """Fleet API via GrpcAdapter servicer."""

    def SendReceive(  # pylint: disable=too-many-return-statements
        self, request: MessageContainer, context: grpc.ServicerContext
    ) -> MessageContainer:
        """."""
        log(DEBUG, "GrpcAdapterServicer.SendReceive")
        if request.grpc_message_name == CreateNodeRequest.__qualname__:
            return _handle(request, context, CreateNodeRequest, self.CreateNode)
        if request.grpc_message_name == DeleteNodeRequest.__qualname__:
            return _handle(request, context, DeleteNodeRequest, self.DeleteNode)
        if request.grpc_message_name == PingRequest.__qualname__:
            return _handle(request, context, PingRequest, self.Ping)
        if request.grpc_message_name == GetRunRequest.__qualname__:
            return _handle(request, context, GetRunRequest, self.GetRun)
        if request.grpc_message_name == GetFabRequest.__qualname__:
            return _handle(request, context, GetFabRequest, self.GetFab)
        if request.grpc_message_name == PullMessagesRequest.__qualname__:
            return _handle(request, context, PullMessagesRequest, self.PullMessages)
        if request.grpc_message_name == PushMessagesRequest.__qualname__:
            return _handle(request, context, PushMessagesRequest, self.PushMessages)
        raise ValueError(f"Invalid grpc_message_name: {request.grpc_message_name}")

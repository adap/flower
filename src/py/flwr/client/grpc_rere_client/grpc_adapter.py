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
"""GrpcAdapter implementation."""


import sys
from logging import DEBUG
from typing import Any, TypeVar, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import log
from flwr.common.constant import (
    GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY,
    GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY,
)
from flwr.common.version import package_version
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    DeleteNodeResponse,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611

T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapter:
    """Adapter class to send and receive gRPC messages via the ``GrpcAdapterStub``.

    This class utilizes the ``GrpcAdapterStub`` to send and receive gRPC messages
    which are defined and used by the Fleet API, as defined in ``fleet.proto``.
    """

    def __init__(self, channel: grpc.Channel) -> None:
        self.stub = GrpcAdapterStub(channel)

    def _send_and_receive(
        self, request: GrpcMessage, response_type: type[T], **kwargs: Any
    ) -> T:
        # Serialize request
        container_req = MessageContainer(
            metadata={GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY: package_version},
            grpc_message_name=request.__class__.__qualname__,
            grpc_message_content=request.SerializeToString(),
        )

        # Send via the stub
        container_res = cast(
            MessageContainer, self.stub.SendReceive(container_req, **kwargs)
        )

        # Handle control message
        should_exit = (
            container_res.metadata.get(GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY, "false")
            == "true"
        )
        if should_exit:
            log(
                DEBUG,
                'Received shutdown signal: exit flag is set to ``"true"``. Exiting...',
            )
            sys.exit(0)

        # Check the grpc_message_name of the response
        if container_res.grpc_message_name != response_type.__qualname__:
            raise ValueError(
                f"Invalid grpc_message_name. Expected {response_type.__qualname__}"
                f", but got {container_res.grpc_message_name}."
            )

        # Deserialize response
        response = response_type()
        response.ParseFromString(container_res.grpc_message_content)
        return response

    def CreateNode(  # pylint: disable=C0103
        self, request: CreateNodeRequest, **kwargs: Any
    ) -> CreateNodeResponse:
        """."""
        return self._send_and_receive(request, CreateNodeResponse, **kwargs)

    def DeleteNode(  # pylint: disable=C0103
        self, request: DeleteNodeRequest, **kwargs: Any
    ) -> DeleteNodeResponse:
        """."""
        return self._send_and_receive(request, DeleteNodeResponse, **kwargs)

    def Ping(  # pylint: disable=C0103
        self, request: PingRequest, **kwargs: Any
    ) -> PingResponse:
        """."""
        return self._send_and_receive(request, PingResponse, **kwargs)

    def PullTaskIns(  # pylint: disable=C0103
        self, request: PullTaskInsRequest, **kwargs: Any
    ) -> PullTaskInsResponse:
        """."""
        return self._send_and_receive(request, PullTaskInsResponse, **kwargs)

    def PushTaskRes(  # pylint: disable=C0103
        self, request: PushTaskResRequest, **kwargs: Any
    ) -> PushTaskResResponse:
        """."""
        return self._send_and_receive(request, PushTaskResResponse, **kwargs)

    def GetRun(  # pylint: disable=C0103
        self, request: GetRunRequest, **kwargs: Any
    ) -> GetRunResponse:
        """."""
        return self._send_and_receive(request, GetRunResponse, **kwargs)

    def GetFab(  # pylint: disable=C0103
        self, request: GetFabRequest, **kwargs: Any
    ) -> GetFabResponse:
        """."""
        return self._send_and_receive(request, GetFabResponse, **kwargs)

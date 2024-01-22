# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Flower driver service client."""


from logging import ERROR, INFO, WARNING
from typing import Optional

import grpc

from flwr.common import EventType, event
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.driver_pb2_grpc import DriverStub  # pylint: disable=E0611

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `GrpcDriver` instance before calling any of the other
`GrpcDriver` methods.
"""


class GrpcDriver:
    """`GrpcDriver` provides access to the gRPC Driver API/service."""

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self.driver_service_address = driver_service_address
        self.root_certificates = root_certificates
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[DriverStub] = None

    def connect(self) -> None:
        """Connect to the Driver API."""
        event(EventType.DRIVER_CONNECT)
        if self.channel is not None or self.stub is not None:
            log(WARNING, "Already connected")
            return
        self.channel = create_channel(
            server_address=self.driver_service_address,
            insecure=(self.root_certificates is None),
            root_certificates=self.root_certificates,
        )
        self.stub = DriverStub(self.channel)
        log(INFO, "[Driver] Connected to %s", self.driver_service_address)

    def disconnect(self) -> None:
        """Disconnect from the Driver API."""
        event(EventType.DRIVER_DISCONNECT)
        if self.channel is None or self.stub is None:
            log(WARNING, "Already disconnected")
            return
        channel = self.channel
        self.channel = None
        self.stub = None
        channel.close()
        log(INFO, "[Driver] Disconnected")

    def create_run(self, req: CreateRunRequest) -> CreateRunResponse:
        """Request for run ID."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriver` instance not connected")

        # Call Driver API
        res: CreateRunResponse = self.stub.CreateRun(request=req)
        return res

    def get_nodes(self, req: GetNodesRequest) -> GetNodesResponse:
        """Get client IDs."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriver` instance not connected")

        # Call gRPC Driver API
        res: GetNodesResponse = self.stub.GetNodes(request=req)
        return res

    def push_task_ins(self, req: PushTaskInsRequest) -> PushTaskInsResponse:
        """Schedule tasks."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriver` instance not connected")

        # Call gRPC Driver API
        res: PushTaskInsResponse = self.stub.PushTaskIns(request=req)
        return res

    def pull_task_res(self, req: PullTaskResRequest) -> PullTaskResResponse:
        """Get task results."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriver` instance not connected")

        # Call Driver API
        res: PullTaskResResponse = self.stub.PullTaskRes(request=req)
        return res

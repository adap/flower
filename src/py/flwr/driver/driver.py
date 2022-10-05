# Copyright 2022 Adap GmbH. All Rights Reserved.
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


from logging import INFO, WARNING
from typing import Optional

import grpc

from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.typing import ClientMessage
from flwr.proto import driver_pb2_grpc

from .messages import (
    CreateTasksRequest,
    CreateTasksResponse,
    GetClientsRequest,
    GetClientsResponse,
    GetResultsRequest,
    GetResultsResponse,
    Result,
)

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"


class Driver:
    """`Driver` provides access to the Driver API."""

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        certificates: Optional[bytes] = None,
    ) -> None:
        self.driver_service_address = driver_service_address
        self.certificates = certificates
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[driver_pb2_grpc.DriverStub] = None

    def connect(self) -> None:
        """Connect to the Driver API."""
        if self.channel is None or self.stub is None:
            log(WARNING, "Already connected")
            return
        self.channel = create_channel(
            server_address=self.driver_service_address,
            root_certificates=self.certificates,
        )
        self.stub = driver_pb2_grpc.DriverStub(self.channel)
        log(INFO, "[Driver] Connected")

    def disconnect(self) -> None:
        """Disconnect from the Driver API."""
        if self.channel is None or self.stub is None:
            log(WARNING, "Already disconnected")
            return
        channel = self.channel
        self.channel = None
        self.stub = None
        channel.close()
        log(INFO, "[Driver] Disconnected")

    def get_clients(self, req: GetClientsRequest) -> GetClientsResponse:
        """."""
        # pylint: disable=no-self-use,unused-argument
        # [...] call DriverAPI
        return GetClientsResponse(client_ids=list(range(5)))

    def create_tasks(self, req: CreateTasksRequest) -> CreateTasksResponse:
        """."""
        # pylint: disable=no-self-use
        # [...] call DriverAPI
        num_tasks: int = sum([len(ta.client_ids) for ta in req.task_assignments])
        return CreateTasksResponse(task_ids=list(range(num_tasks)))

    def get_results(self, req: GetResultsRequest) -> GetResultsResponse:
        """."""
        # pylint: disable=no-self-use
        # [...] call DriverAPI
        return GetResultsResponse(
            results=[
                Result(task_id=task_id, legacy_client_message=ClientMessage())
                for task_id in req.task_ids
            ]
        )

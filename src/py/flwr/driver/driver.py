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


from logging import ERROR, INFO, WARNING
from typing import Optional

import grpc

from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.driver import serde
from flwr.proto import driver_pb2, driver_pb2_grpc

from .messages import (
    CreateTasksRequest,
    CreateTasksResponse,
    GetClientsRequest,
    GetClientsResponse,
    GetResultsRequest,
    GetResultsResponse,
)

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `Driver` instance before calling any of the other `Driver`
methods.
"""


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
        if self.channel is not None or self.stub is not None:
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
        """Get client IDs."""
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Serialize, call Driver API, deserialize
        req_proto = serde.get_clients_request_to_proto(req)
        res_proto: driver_pb2.GetClientsResponse = self.stub.GetClients(
            request=req_proto
        )
        return serde.get_clients_response_from_proto(res_proto)

    def create_tasks(self, req: CreateTasksRequest) -> CreateTasksResponse:
        """Schedule tasks."""
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Serialize, call Driver API, deserialize
        req_proto = serde.create_tasks_request_to_proto(req)
        res_proto: driver_pb2.CreateTasksResponse = self.stub.CreateTasks(
            request=req_proto
        )
        return serde.create_tasks_response_from_proto(res_proto)

    def get_results(self, req: GetResultsRequest) -> GetResultsResponse:
        """Get task results."""
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Serialize, call Driver API, deserialize
        req_proto = serde.get_results_request_to_proto(req)
        res_proto: driver_pb2.GetResultsResponse = self.stub.GetResults(
            request=req_proto
        )
        return serde.get_results_response_from_proto(res_proto)

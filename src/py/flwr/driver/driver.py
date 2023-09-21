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


import time
from logging import ERROR, INFO, WARNING
from typing import Callable, Optional, Union, cast

import grpc

from flwr.common import EventType, event
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.proto.driver_pb2 import (
    CreateWorkloadRequest,
    CreateWorkloadResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.driver_pb2_grpc import DriverStub

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

INITIAL_RETRY_INTERVAL = 1.0
RETRY_TIMES = 3

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `Driver` instance before calling any of the other `Driver`
methods.
"""

WARNING_MESSAGE_SERVICE_UNAVAILABLE = (
    f"[Driver] Service unavailable, retrying %s/{RETRY_TIMES} after %s seconds..."
)

DriverRequest = Union[
    CreateWorkloadRequest,
    GetNodesRequest,
    PushTaskInsRequest,
    PullTaskResRequest,
]
DriverResponse = Union[
    CreateWorkloadResponse,
    GetNodesResponse,
    PushTaskInsResponse,
    PullTaskResResponse,
]


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
        self.stub: Optional[DriverStub] = None

    def connect(self) -> None:
        """Connect to the Driver API."""
        event(EventType.DRIVER_CONNECT)
        if self.channel is not None or self.stub is not None:
            log(WARNING, "Already connected")
            return
        self.channel = create_channel(
            server_address=self.driver_service_address,
            root_certificates=self.certificates,
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

    def create_workload(self, req: CreateWorkloadRequest) -> CreateWorkloadResponse:
        """Request for workload ID."""
        res = cast(
            CreateWorkloadResponse,
            self._try_call_api("CreateWorkload", req),
        )
        return res

    def get_nodes(self, req: GetNodesRequest) -> GetNodesResponse:
        """Get client IDs."""
        res = cast(GetNodesResponse, self._try_call_api("GetNodes", req))
        return res

    def push_task_ins(self, req: PushTaskInsRequest) -> PushTaskInsResponse:
        """Schedule tasks."""
        res = cast(
            PushTaskInsResponse,
            self._try_call_api("PushTaskIns", req),
        )
        return res

    def pull_task_res(self, req: PullTaskResRequest) -> PullTaskResResponse:
        """Get task results."""
        res = cast(
            PullTaskResResponse,
            self._try_call_api("PullTaskRes", req),
        )
        return res

    def _try_call_api(
        self,
        api_name: str,
        request: DriverRequest,
    ) -> DriverResponse:
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise Exception("`Driver` instance not connected")

        # Call Driver API
        api = cast(
            Callable[[DriverRequest], DriverResponse], getattr(self.stub, api_name)
        )
        retry_interval = INITIAL_RETRY_INTERVAL

        for retry_count in range(RETRY_TIMES + 1):
            try:
                return api(request)
            except grpc.RpcError as err:
                if retry_count < RETRY_TIMES:
                    # pylint: disable-next=no-member
                    if err.code() == grpc.StatusCode.UNAVAILABLE:
                        log(
                            WARNING,
                            WARNING_MESSAGE_SERVICE_UNAVAILABLE,
                            retry_count + 1,
                            retry_interval,
                        )

                    time.sleep(retry_interval)
                    # Double the retry interval for exponential backoff
                    retry_interval *= 2
                else:
                    raise
        raise RuntimeError()

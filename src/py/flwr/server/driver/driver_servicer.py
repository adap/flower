# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Driver API servicer."""


from logging import INFO
from typing import Set

import grpc

from flwr.common.logger import log
from flwr.proto import driver_pb2_grpc
from flwr.proto.driver_pb2 import (
    CreateTasksRequest,
    CreateTasksResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetResultsRequest,
    GetResultsResponse,
)
from flwr.server.driver.driver_client_manager import DriverClientManager


class DriverServicer(driver_pb2_grpc.DriverServicer):
    """Driver API servicer."""

    def __init__(
        self,
        driver_client_manager: DriverClientManager,
    ) -> None:
        self.driver_client_manager = driver_client_manager

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        log(INFO, "DriverServicer.GetNodes")
        all_ids: Set[int] = self.driver_client_manager.all_ids()
        return GetNodesResponse(node_ids=list(all_ids))

    def CreateTasks(
        self, request: CreateTasksRequest, context: grpc.ServicerContext
    ) -> CreateTasksResponse:
        log(INFO, "DriverServicer.CreateTasks")
        return CreateTasksResponse(task_ids=[])

    def GetResults(
        self, request: GetResultsRequest, context: grpc.ServicerContext
    ) -> GetResultsResponse:
        log(INFO, "DriverServicer.GetResults")
        return GetResultsResponse(results=[])

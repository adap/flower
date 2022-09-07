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
"""Fleet API servicer."""


from logging import INFO

import grpc

from flwr.common.logger import log
from flwr.proto import fleet_pb2_grpc
from flwr.proto.fleet_pb2 import (
    CreateResultsRequest,
    CreateResultsResponse,
    GetTasksRequest,
    GetTasksResponse,
)


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def GetTasks(
        self, request: GetTasksRequest, context: grpc.ServicerContext
    ) -> GetTasksResponse:
        log(INFO, "GetTasks")
        return super().GetTasks(request, context)

    def CreateResults(
        self, request: CreateResultsRequest, context: grpc.ServicerContext
    ) -> CreateResultsResponse:
        log(INFO, "CreateResults")
        return super().CreateResults(request, context)

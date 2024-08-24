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
"""Fleet API gRPC request-response servicer."""


from logging import DEBUG, INFO

import grpc

from flwr.common.logger import log
from flwr.proto import fleet_pb2_grpc  # pylint: disable=E0611
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
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.state import StateFactory


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def __init__(self, state_factory: StateFactory, ffs_factory: FfsFactory) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory

    def CreateNode(
        self, request: CreateNodeRequest, context: grpc.ServicerContext
    ) -> CreateNodeResponse:
        """."""
        log(INFO, "[Fleet.CreateNode] Request ping_interval=%s", request.ping_interval)
        log(DEBUG, "[Fleet.CreateNode] Request: %s", request)
        response = message_handler.create_node(
            request=request,
            state=self.state_factory.state(),
        )
        log(INFO, "[Fleet.CreateNode] Created node_id=%s", response.node.node_id)
        log(DEBUG, "[Fleet.CreateNode] Response: %s", response)
        return response

    def DeleteNode(
        self, request: DeleteNodeRequest, context: grpc.ServicerContext
    ) -> DeleteNodeResponse:
        """."""
        log(INFO, "[Fleet.DeleteNode] Delete node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.DeleteNode] Request: %s", request)
        return message_handler.delete_node(
            request=request,
            state=self.state_factory.state(),
        )

    def Ping(self, request: PingRequest, context: grpc.ServicerContext) -> PingResponse:
        """."""
        log(DEBUG, "[Fleet.Ping] Request: %s", request)
        return message_handler.ping(
            request=request,
            state=self.state_factory.state(),
        )

    def PullTaskIns(
        self, request: PullTaskInsRequest, context: grpc.ServicerContext
    ) -> PullTaskInsResponse:
        """Pull TaskIns."""
        log(INFO, "[Fleet.PullTaskIns] node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.PullTaskIns] Request: %s", request)
        return message_handler.pull_task_ins(
            request=request,
            state=self.state_factory.state(),
        )

    def PushTaskRes(
        self, request: PushTaskResRequest, context: grpc.ServicerContext
    ) -> PushTaskResResponse:
        """Push TaskRes."""
        if request.task_res_list:
            log(
                INFO,
                "[Fleet.PushTaskRes] Push results from node_id=%s",
                request.task_res_list[0].task.producer.node_id,
            )
        else:
            log(INFO, "[Fleet.PushTaskRes] No task results to push")
        return message_handler.push_task_res(
            request=request,
            state=self.state_factory.state(),
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(INFO, "[Fleet.GetRun] Requesting `Run` for run_id=%s", request.run_id)
        return message_handler.get_run(
            request=request,
            state=self.state_factory.state(),
        )

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB."""
        log(INFO, "[Fleet.GetFab] Requesting FAB for fab_hash=%s", request.hash_str)
        return message_handler.get_fab(
            request=request,
            ffs=self.ffs_factory.ffs(),
        )

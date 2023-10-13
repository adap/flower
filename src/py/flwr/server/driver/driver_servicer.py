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
"""Driver API servicer."""


from logging import INFO
from typing import List, Optional, Set
from uuid import UUID

import grpc

from flwr.common.logger import log
from flwr.proto import driver_pb2_grpc
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
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import TaskRes
from flwr.server.state import State, StateFactory
from flwr.server.utils.validator import validate_task_ins_or_res


class DriverServicer(driver_pb2_grpc.DriverServicer):
    """Driver API servicer."""

    def __init__(self, state_factory: StateFactory) -> None:
        self.state_factory = state_factory

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(INFO, "DriverServicer.GetNodes")
        state: State = self.state_factory.state()
        all_ids: Set[int] = state.get_nodes(request.workload_id)
        nodes: List[Node] = [
            Node(node_id=node_id, anonymous=False) for node_id in all_ids
        ]
        return GetNodesResponse(nodes=nodes)

    def CreateWorkload(
        self, request: CreateWorkloadRequest, context: grpc.ServicerContext
    ) -> CreateWorkloadResponse:
        """Create workload ID."""
        log(INFO, "DriverServicer.CreateWorkload")
        state: State = self.state_factory.state()
        workload_id = state.create_workload()
        return CreateWorkloadResponse(workload_id=workload_id)

    def PushTaskIns(
        self, request: PushTaskInsRequest, context: grpc.ServicerContext
    ) -> PushTaskInsResponse:
        """Push a set of TaskIns."""
        log(INFO, "DriverServicer.PushTaskIns")

        # Validate request
        _raise_if(len(request.task_ins_list) == 0, "`task_ins_list` must not be empty")
        for task_ins in request.task_ins_list:
            validation_errors = validate_task_ins_or_res(task_ins)
            _raise_if(bool(validation_errors), ", ".join(validation_errors))

        # Init state
        state: State = self.state_factory.state()

        # Store each TaskIns
        task_ids: List[Optional[UUID]] = []
        for task_ins in request.task_ins_list:
            task_id: Optional[UUID] = state.store_task_ins(task_ins=task_ins)
            task_ids.append(task_id)

        return PushTaskInsResponse(
            task_ids=[str(task_id) if task_id else "" for task_id in task_ids]
        )

    def PullTaskRes(
        self, request: PullTaskResRequest, context: grpc.ServicerContext
    ) -> PullTaskResResponse:
        """Pull a set of TaskRes."""
        log(INFO, "DriverServicer.PullTaskRes")

        # Convert each task_id str to UUID
        task_ids: Set[UUID] = {UUID(task_id) for task_id in request.task_ids}

        # Init state
        state: State = self.state_factory.state()

        # Register callback
        def on_rpc_done() -> None:
            log(INFO, "DriverServicer.PullTaskRes callback: delete TaskIns/TaskRes")

            if context.is_active():
                return
            if context.code() != grpc.StatusCode.OK:
                return

            # Delete delivered TaskIns and TaskRes
            state.delete_tasks(task_ids=task_ids)

        context.add_callback(on_rpc_done)

        # Read from state
        task_res_list: List[TaskRes] = state.get_task_res(task_ids=task_ids, limit=None)

        context.set_code(grpc.StatusCode.OK)
        return PullTaskResResponse(task_res_list=task_res_list)


def _raise_if(validation_error: bool, detail: str) -> None:
    if validation_error:
        raise ValueError(f"Malformed PushTaskInsRequest: {detail}")

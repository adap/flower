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
from typing import List, Optional, Set
from uuid import UUID

import grpc

from flwr.common.logger import log
from flwr.proto import driver_pb2_grpc
from flwr.proto.driver_pb2 import (
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.task_pb2 import TaskRes
from flwr.server.driver.driver_client_manager import DriverClientManager
from flwr.server.driver.state import DriverState


class DriverServicer(driver_pb2_grpc.DriverServicer):
    """Driver API servicer."""

    def __init__(
        self,
        driver_client_manager: DriverClientManager,
        driver_state: DriverState,
    ) -> None:
        self.driver_client_manager = driver_client_manager
        self.driver_state = driver_state

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(INFO, "DriverServicer.GetNodes")
        all_ids: Set[int] = self.driver_client_manager.all_ids()
        return GetNodesResponse(node_ids=list(all_ids))

    def PushTaskIns(
        self, request: PushTaskInsRequest, context: grpc.ServicerContext
    ) -> PushTaskInsResponse:
        """Push a set of TaskIns."""
        log(INFO, "DriverServicer.PushTaskIns")

        task_ids: List[Optional[UUID]] = []
        for task_ins in request.task_ins_set:
            # TODO Verify that task_id is not set
            # TODO Verify that only legacy_server_message is set

            # Store TaskIns
            task_id: Optional[UUID] = self.driver_state.store_task_ins(
                task_ins=task_ins
            )
            task_ids.append(task_id)

        return PushTaskInsResponse(task_ids=[str(task_id) for task_id in task_ids])

    def PullTaskRes(
        self, request: PullTaskResRequest, context: grpc.ServicerContext
    ) -> PullTaskResResponse:
        """Pull a set of TaskRes."""
        log(INFO, "DriverServicer.PullTaskRes")

        # Convert each task_id str to UUID
        task_ids: Set[UUID] = {UUID(task_id) for task_id in request.task_ids}

        # Read from state
        task_res_set: List[TaskRes] = self.driver_state.get_task_res(
            node_id=None, task_ids=task_ids, limit=10
        )

        return PullTaskResResponse(task_res_set=task_res_set)

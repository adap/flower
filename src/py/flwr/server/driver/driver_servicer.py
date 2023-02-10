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
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.server.state import State


class DriverServicer(driver_pb2_grpc.DriverServicer):
    """Driver API servicer."""

    def __init__(
        self,
        state: State,
    ) -> None:
        self.state = state

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(INFO, "DriverServicer.GetNodes")
        all_ids: Set[int] = self.state.get_nodes()
        return GetNodesResponse(node_ids=list(all_ids))

    def PushTaskIns(
        self, request: PushTaskInsRequest, context: grpc.ServicerContext
    ) -> PushTaskInsResponse:
        """Push a set of TaskIns."""
        log(INFO, "DriverServicer.PushTaskIns")

        # Validate request
        _raise_if(len(request.task_ins_list) == 0, "`task_ins_list` must not be empty")
        for task_ins in request.task_ins_list:
            _validate_incoming_task_ins(task_ins=task_ins)

        # Store each TaskIns
        task_ids: List[Optional[UUID]] = []
        for task_ins in request.task_ins_list:
            task_id: Optional[UUID] = self.state.store_task_ins(task_ins=task_ins)
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

        # Read from state
        task_res_list: List[TaskRes] = self.state.get_task_res(
            task_ids=task_ids, limit=None
        )
        return PullTaskResResponse(task_res_list=task_res_list)


def _validate_incoming_task_ins(task_ins: TaskIns) -> None:
    """Validate incoming TaskIns."""

    _raise_if(task_ins.task_id != "", "non-empty `task_id`")
    _raise_if(not task_ins.HasField("task"), "`task` does not set field `task`")

    task: Task = task_ins.task

    # Task producer
    _raise_if(not task.HasField("producer"), "`producer` does not set field `producer`")
    _raise_if(task.producer.node_id != 0, "`producer.node_id` is not 0")
    _raise_if(not task.producer.anonymous, "`producer` is not anonymous")

    # Task consumer
    _raise_if(not task.HasField("consumer"), "`consumer` does not set field `consumer`")
    _raise_if(
        task.consumer.anonymous and task.consumer.node_id != 0,
        "anonymous consumers MUST NOT set a `node_id`",
    )
    _raise_if(
        not task.consumer.anonymous and task.consumer.node_id == 0,
        "non-anonymous consumer MUST provide a `node_id`",
    )

    # Created/delivered/TTL
    _raise_if(task.created_at != "", "`created_at` must be an empty str")
    _raise_if(task.delivered_at != "", "`delivered_at` must be an empty str")
    _raise_if(task.ttl != "", "`ttl` must be an empty str")

    # Legacy ServerMessage/ClientMessage
    _raise_if(
        task.HasField("legacy_client_message"),
        "`legacy_client_message` is not `None`",
    )
    _raise_if(
        not task.HasField("legacy_server_message"),
        "`task` does not set field `legacy_server_message`",
    )
    _raise_if(
        not task.legacy_server_message.HasField("msg"),
        "`legacy_server_message` does not set field `msg`",
    )

    # Ancestors
    _raise_if(len(task.ancestry) != 0, "`ancestry` is not empty")


def _raise_if(validation_error: bool, detail: str) -> None:
    if validation_error:
        raise ValueError(f"Malformed PushTaskInsRequest: {detail}")

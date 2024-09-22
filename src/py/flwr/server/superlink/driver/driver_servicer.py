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
"""Driver API servicer."""


import time
from logging import DEBUG
from typing import Optional
from uuid import UUID

import grpc

from flwr.common.logger import log
from flwr.common.serde import (
    fab_from_proto,
    fab_to_proto,
    user_config_from_proto,
    user_config_to_proto,
)
from flwr.common.typing import Fab
from flwr.proto import driver_pb2_grpc  # pylint: disable=E0611
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
    GetRunRequest,
    GetRunResponse,
    Run,
)
from flwr.proto.task_pb2 import TaskRes  # pylint: disable=E0611
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.state import State, StateFactory
from flwr.server.utils.validator import validate_task_ins_or_res


class DriverServicer(driver_pb2_grpc.DriverServicer):
    """Driver API servicer."""

    def __init__(self, state_factory: StateFactory, ffs_factory: FfsFactory) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "DriverServicer.GetNodes")
        state: State = self.state_factory.state()
        all_ids: set[int] = state.get_nodes(request.run_id)
        nodes: list[Node] = [
            Node(node_id=node_id, anonymous=False) for node_id in all_ids
        ]
        return GetNodesResponse(nodes=nodes)

    def CreateRun(
        self, request: CreateRunRequest, context: grpc.ServicerContext
    ) -> CreateRunResponse:
        """Create run ID."""
        log(DEBUG, "DriverServicer.CreateRun")
        state: State = self.state_factory.state()
        if request.HasField("fab"):
            fab = fab_from_proto(request.fab)
            ffs: Ffs = self.ffs_factory.ffs()
            fab_hash = ffs.put(fab.content, {})
            _raise_if(
                fab_hash != fab.hash_str,
                f"FAB ({fab.hash_str}) hash from request doesn't match contents",
            )
        else:
            fab_hash = ""
        run_id = state.create_run(
            request.fab_id,
            request.fab_version,
            fab_hash,
            user_config_from_proto(request.override_config),
        )
        return CreateRunResponse(run_id=run_id)

    def PushTaskIns(
        self, request: PushTaskInsRequest, context: grpc.ServicerContext
    ) -> PushTaskInsResponse:
        """Push a set of TaskIns."""
        log(DEBUG, "DriverServicer.PushTaskIns")

        # Set pushed_at (timestamp in seconds)
        pushed_at = time.time()
        for task_ins in request.task_ins_list:
            task_ins.task.pushed_at = pushed_at

        # Validate request
        _raise_if(len(request.task_ins_list) == 0, "`task_ins_list` must not be empty")
        for task_ins in request.task_ins_list:
            validation_errors = validate_task_ins_or_res(task_ins)
            _raise_if(bool(validation_errors), ", ".join(validation_errors))

        # Init state
        state: State = self.state_factory.state()

        # Store each TaskIns
        task_ids: list[Optional[UUID]] = []
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
        log(DEBUG, "DriverServicer.PullTaskRes")

        # Convert each task_id str to UUID
        task_ids: set[UUID] = {UUID(task_id) for task_id in request.task_ids}

        # Init state
        state: State = self.state_factory.state()

        # Register callback
        def on_rpc_done() -> None:
            log(DEBUG, "DriverServicer.PullTaskRes callback: delete TaskIns/TaskRes")

            if context.is_active():
                return
            if context.code() != grpc.StatusCode.OK:
                return

            # Delete delivered TaskIns and TaskRes
            state.delete_tasks(task_ids=task_ids)

        context.add_callback(on_rpc_done)

        # Read from state
        task_res_list: list[TaskRes] = state.get_task_res(task_ids=task_ids, limit=None)

        context.set_code(grpc.StatusCode.OK)
        return PullTaskResResponse(task_res_list=task_res_list)

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "DriverServicer.GetRun")

        # Init state
        state: State = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(
            run=Run(
                run_id=run.run_id,
                fab_id=run.fab_id,
                fab_version=run.fab_version,
                override_config=user_config_to_proto(run.override_config),
                fab_hash=run.fab_hash,
            )
        )

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB from Ffs."""
        log(DEBUG, "DriverServicer.GetFab")

        ffs: Ffs = self.ffs_factory.ffs()
        if result := ffs.get(request.hash_str):
            fab = Fab(request.hash_str, result[0])
            return GetFabResponse(fab=fab_to_proto(fab))

        raise ValueError(f"Found no FAB with hash: {request.hash_str}")


def _raise_if(validation_error: bool, detail: str) -> None:
    if validation_error:
        raise ValueError(f"Malformed PushTaskInsRequest: {detail}")

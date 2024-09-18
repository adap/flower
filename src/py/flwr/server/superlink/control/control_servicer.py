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


from logging import DEBUG

import grpc

from flwr.common.constant import RunStatus
from flwr.common.logger import log
from flwr.common.serde import fab_from_proto, user_config_from_proto
from flwr.common.typing import StatusInfo
from flwr.proto import control_pb2_grpc  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
    GetRunStatusRequest,
    GetRunStatusResponse,
)
from flwr.proto.run_pb2 import StatusInfo as StatusInfoProto  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.server.superlink.ffs.ffs import Ffs
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.state import State, StateFactory


class ControlServicer(control_pb2_grpc.ControlServicer):
    """Control API servicer."""

    def __init__(self, state_factory: StateFactory, ffs_factory: FfsFactory) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory

    def CreateRun(
        self, request: CreateRunRequest, context: grpc.ServicerContext
    ) -> CreateRunResponse:
        """Create run ID."""
        rpc_name = "ControlServicer.CreateRun"
        log(DEBUG, rpc_name)
        state: State = self.state_factory.state()
        if request.HasField("fab"):
            fab = fab_from_proto(request.fab)
            ffs: Ffs = self.ffs_factory.ffs()
            fab_hash = ffs.put(fab.content, {})
            if fab_hash != fab.hash_str:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"{rpc_name}: FAB ({fab.hash_str}) hash from request "
                    "doesn't match contents",
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

    def GetRunStatus(
        self, request: GetRunStatusRequest, context: grpc.ServicerContext
    ) -> GetRunStatusResponse:
        """Get the status of a run."""
        log(DEBUG, "ControlServicer.GetRunStatus")
        state = self.state_factory.state()

        # Get run statuses
        status_dict = state.get_run_status(set(request.run_ids))

        # Serialize the statuses and return
        status_dict_proto = {
            k: StatusInfoProto(**vars(v)) for k, v in status_dict.items()
        }
        return GetRunStatusResponse(run_status_dict=status_dict_proto)

    def UpdateRunStatus(
        self, request: UpdateRunStatusRequest, context: grpc.ServicerContext
    ) -> UpdateRunStatusResponse:
        """Update the status of a run.

        Notes
        -----
        Currently, the `UpdateRunStatus` RPC in the Control service is only invoked
        by the SuperExec to signal the completion of a run.
        """
        rpc_name = "ControlServicer.UpdateRunStatus"
        log(DEBUG, rpc_name)
        state = self.state_factory.state()

        # Retrieve the run ID
        run_id = request.run_id

        # Deserialize the status info
        proto = request.info
        info = StatusInfo(proto.status, proto.sub_status, proto.reason)
        if info.status != RunStatus.FINISHED:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{rpc_name}: This RPC can only be used to signal the "
                "completion of a run.",
            )

        # Check the current status
        status_dict = state.get_run_status({run_id})
        if not status_dict:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"{rpc_name}: `run_id` was not found."
            )
        current_info = status_dict[run_id]

        # Check if the current status is RunStatus.FINISHED
        if current_info.status != RunStatus.FINISHED:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"{rpc_name}: The current status of the run is not RunStatus.FINISHED. "
                "This means that the ServerApp failed to report the run's "
                "completion as expected.",
            )

        return UpdateRunStatusResponse()

# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""SimulationIo API servicer."""


import threading
from logging import DEBUG, INFO

import grpc
from grpc import ServicerContext

from flwr.common.constant import Status
from flwr.common.logger import log
from flwr.common.serde import (
    config_record_to_proto,
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    run_status_from_proto,
    run_status_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab, RunStatus
from flwr.proto import simulationio_pb2_grpc
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsRequest,
    PullAppInputsResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetFederationOptionsRequest,
    GetFederationOptionsResponse,
    GetRunStatusRequest,
    GetRunStatusResponse,
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.utils import abort_if
from flwr.supercore.ffs import FfsFactory


class SimulationIoServicer(simulationio_pb2_grpc.SimulationIoServicer):
    """SimulationIo API servicer."""

    def __init__(
        self, state_factory: LinkStateFactory, ffs_factory: FfsFactory
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.lock = threading.RLock()

    def ListAppsToLaunch(
        self,
        request: ListAppsToLaunchRequest,
        context: grpc.ServicerContext,
    ) -> ListAppsToLaunchResponse:
        """Get run IDs with pending messages."""
        log(DEBUG, "SimulationIoServicer.ListAppsToLaunch")

        # Initialize state connection
        state = self.state_factory.state()

        # Get IDs of runs in pending status
        run_ids = state.get_run_ids(flwr_aid=None)
        pending_run_ids = []
        for run_id, status in state.get_run_status(run_ids).items():
            if status.status == Status.PENDING:
                pending_run_ids.append(run_id)

        # Return run IDs
        return ListAppsToLaunchResponse(run_ids=pending_run_ids)

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "SimulationIoServicer.RequestToken")

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        token = state.create_token(request.run_id)

        # Return the token
        return RequestTokenResponse(token=token or "")

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: ServicerContext
    ) -> PullAppInputsResponse:
        """Pull SimultionIo process inputs."""
        log(DEBUG, "SimultionIoServicer.SimultionIoInputs")
        # Init access to LinkState and Ffs
        state = self.state_factory.state()
        ffs = self.ffs_factory.ffs()

        # Lock access to LinkState, preventing obtaining the same pending run_id
        with self.lock:
            # Attempt getting the run_id of a pending run
            run_id = state.get_pending_run_id()
            # If there's no pending run, return an empty response
            if run_id is None:
                return PullAppInputsResponse()

            # Retrieve Context, Run and Fab for the run_id
            serverapp_ctxt = state.get_serverapp_context(run_id)
            run = state.get_run(run_id)
            fab = None
            if run and run.fab_hash:
                if result := ffs.get(run.fab_hash):
                    fab = Fab(run.fab_hash, result[0])
            if run and fab and serverapp_ctxt:
                # Update run status to STARTING
                if state.update_run_status(run_id, RunStatus(Status.STARTING, "", "")):
                    log(INFO, "Starting run %d", run_id)
                    return PullAppInputsResponse(
                        context=context_to_proto(serverapp_ctxt),
                        run=run_to_proto(run),
                        fab=fab_to_proto(fab),
                    )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to STARTING
        raise RuntimeError(f"Failed to start run {run_id}")

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: ServicerContext
    ) -> PushAppOutputsResponse:
        """Push Simulation process outputs."""
        log(DEBUG, "SimultionIoServicer.PushAppOutputs")
        state = self.state_factory.state()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            None,
            context,
        )

        state.set_serverapp_context(request.run_id, context_from_proto(request.context))
        return PushAppOutputsResponse()

    def UpdateRunStatus(
        self, request: UpdateRunStatusRequest, context: grpc.ServicerContext
    ) -> UpdateRunStatusResponse:
        """Update the status of a run."""
        log(DEBUG, "SimultionIoServicer.UpdateRunStatus")
        state = self.state_factory.state()

        # Abort if the run is finished
        abort_if(request.run_id, [Status.FINISHED], state, None, context)

        # Update the run status
        state.update_run_status(
            run_id=request.run_id, new_status=run_status_from_proto(request.run_status)
        )
        return UpdateRunStatusResponse()

    def GetRunStatus(
        self, request: GetRunStatusRequest, context: ServicerContext
    ) -> GetRunStatusResponse:
        """Get status of requested runs."""
        log(DEBUG, "SimultionIoServicer.GetRunStatus")
        state = self.state_factory.state()

        statuses = state.get_run_status(set(request.run_ids))

        return GetRunStatusResponse(
            run_status_dict={
                run_id: run_status_to_proto(status)
                for run_id, status in statuses.items()
            }
        )

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        log(DEBUG, "SimultionIoServicer.PushLogs")
        state = self.state_factory.state()

        # Add logs to LinkState
        merged_logs = "".join(request.logs)
        state.add_serverapp_log(request.run_id, merged_logs)
        return PushLogsResponse()

    def GetFederationOptions(
        self, request: GetFederationOptionsRequest, context: ServicerContext
    ) -> GetFederationOptionsResponse:
        """Get Federation Options associated with a run."""
        log(DEBUG, "SimultionIoServicer.GetFederationOptions")
        state = self.state_factory.state()

        federation_options = state.get_federation_options(request.run_id)
        if federation_options is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Expected federation options to be set, but none available.",
            )
            return GetFederationOptionsResponse()
        return GetFederationOptionsResponse(
            federation_options=config_record_to_proto(federation_options)
        )

    def SendAppHeartbeat(
        self, request: SendAppHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendAppHeartbeatResponse:
        """Handle a heartbeat from the ServerApp in simulation."""
        log(DEBUG, "SimultionIoServicer.SendAppHeartbeat")

        # Init state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        # The app heartbeat can only be acknowledged if the run is in
        # starting or running status.
        success = state.acknowledge_app_heartbeat(
            run_id=request.run_id,
            heartbeat_interval=request.heartbeat_interval,
        )

        return SendAppHeartbeatResponse(success=success)

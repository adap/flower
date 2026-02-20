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

from flwr.common.constant import ExecPluginType, Status
from flwr.common.logger import log
from flwr.common.serde import (
    config_record_to_proto,
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    run_status_from_proto,
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
    GetRunRequest,
    GetRunResponse,
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.server.superlink.superexec_auth import (
    SuperExecAuthConfig,
    get_disabled_superexec_auth_config,
    superexec_auth_metadata_present,
    verify_superexec_signed_metadata,
)
from flwr.server.superlink.utils import abort_if
from flwr.supercore.ffs import FfsFactory


class SimulationIoServicer(simulationio_pb2_grpc.SimulationIoServicer):
    """SimulationIo API servicer."""

    _METHOD_LIST_APPS_TO_LAUNCH = "/flwr.proto.SimulationIo/ListAppsToLaunch"
    _METHOD_REQUEST_TOKEN = "/flwr.proto.SimulationIo/RequestToken"
    _METHOD_GET_RUN = "/flwr.proto.SimulationIo/GetRun"

    def __init__(
        self,
        state_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        superexec_auth_config: SuperExecAuthConfig | None = None,
    ) -> None:
        """Initialize SimulationIo servicer dependencies and auth settings."""
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.superexec_auth_config = (
            superexec_auth_config or get_disabled_superexec_auth_config()
        )
        self.lock = threading.RLock()

    def ListAppsToLaunch(
        self,
        request: ListAppsToLaunchRequest,
        context: grpc.ServicerContext,
    ) -> ListAppsToLaunchResponse:
        """Get run IDs with pending messages."""
        log(DEBUG, "SimulationIoServicer.ListAppsToLaunch")
        self._verify_superexec_auth_if_enabled(
            context=context, method=self._METHOD_LIST_APPS_TO_LAUNCH
        )

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
        self._verify_superexec_auth_if_enabled(
            context=context, method=self._METHOD_REQUEST_TOKEN
        )

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        token = state.create_token(request.run_id)

        # Transition the run to STARTING if token creation was successful
        if token:
            state.update_run_status(
                run_id=request.run_id,
                new_status=RunStatus(Status.STARTING, "", ""),
            )

        # Return the token
        return RequestTokenResponse(token=token or "")

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "SimulationIoServicer.GetRun")
        self._verify_get_run_auth_if_enabled(
            request=request, context=context, method=self._METHOD_GET_RUN
        )

        # Init state
        state = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(run))

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: ServicerContext
    ) -> PullAppInputsResponse:
        """Pull SimultionIo process inputs."""
        log(DEBUG, "SimultionIoServicer.SimultionIoInputs")
        # Init access to LinkState and Ffs
        state = self.state_factory.state()
        ffs = self.ffs_factory.ffs()

        # Validate the token
        run_id = self._verify_token(request.token, context)

        # Lock access to LinkState, preventing obtaining the same pending run_id
        with self.lock:
            # Retrieve Context, Run and Fab for the run_id
            serverapp_ctxt = state.get_serverapp_context(run_id)
            run = state.get_run(run_id)
            fab = None
            if run and run.fab_hash:
                if result := ffs.get(run.fab_hash):
                    fab = Fab(run.fab_hash, result[0], result[1])
            if run and fab and serverapp_ctxt:
                # Update run status to RUNNING
                if state.update_run_status(run_id, RunStatus(Status.RUNNING, "", "")):
                    log(INFO, "Starting run %d", run_id)
                    return PullAppInputsResponse(
                        context=context_to_proto(serverapp_ctxt),
                        run=run_to_proto(run),
                        fab=fab_to_proto(fab),
                    )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to RUNNING
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Failed to start run {run_id}",
        )
        raise RuntimeError("Unreachable code")  # for mypy

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: ServicerContext
    ) -> PushAppOutputsResponse:
        """Push Simulation process outputs."""
        log(DEBUG, "SimultionIoServicer.PushAppOutputs")

        # Validate the token
        run_id = self._verify_token(request.token, context)

        # Init access to LinkState
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

        # Remove the token
        state.delete_token(run_id)
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
        """Handle a heartbeat from an app process."""
        log(DEBUG, "SimulationIoServicer.SendAppHeartbeat")

        # Init state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        success = state.acknowledge_app_heartbeat(request.token)
        return SendAppHeartbeatResponse(success=success)

    def _verify_token(self, token: str, context: grpc.ServicerContext) -> int:
        """Verify the token and return the associated run ID."""
        state = self.state_factory.state()
        run_id = state.get_run_id_by_token(token)
        if run_id is None or not state.verify_token(run_id, token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")
        return run_id

    def _verify_token_for_run(
        self, token: str, run_id: int, context: grpc.ServicerContext
    ) -> None:
        """Verify token and ensure it belongs to the provided run ID."""
        token_run_id = self._verify_token(token, context)
        if token_run_id != run_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token for run ID.",
            )
            raise RuntimeError("This line should never be reached.")

    def _verify_superexec_auth_if_enabled(
        self, context: grpc.ServicerContext, method: str
    ) -> None:
        """Verify SuperExec signed metadata when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            return
        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=ExecPluginType.SIMULATION,
            cfg=self.superexec_auth_config,
        )

    def _verify_get_run_auth_if_enabled(
        self, request: GetRunRequest, context: grpc.ServicerContext, method: str
    ) -> None:
        """Authorize GetRun with one mechanism when SuperExec auth is enabled."""
        if not self.superexec_auth_config.enabled:
            # Legacy behavior by design: when SuperExec auth is disabled, GetRun
            # remains unauthenticated and tokenless requests are allowed.
            return

        token_present = bool(request.token)
        signed_metadata_present = superexec_auth_metadata_present(context)
        if token_present == signed_metadata_present:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Exactly one authentication mechanism must be provided.",
            )
            raise RuntimeError("This line should never be reached.")

        if token_present:
            self._verify_token_for_run(request.token, request.run_id, context)
            return

        verify_superexec_signed_metadata(
            context=context,
            method=method,
            plugin_type=ExecPluginType.SIMULATION,
            cfg=self.superexec_auth_config,
        )

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
"""SuperExec API servicer."""


import time
from collections.abc import Generator
from logging import ERROR, INFO
from typing import Any

import grpc

from flwr.common import now
from flwr.common.constant import LOG_STREAM_INTERVAL, Status
from flwr.common.logger import log
from flwr.common.serde import (
    configs_record_from_proto,
    run_status_to_proto,
    run_to_proto,
    scalar_from_proto,
    user_config_from_proto,
)
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    ListRequest,
    ListResponse,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

from .executor import Executor


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """SuperExec API servicer."""

    def __init__(
        self,
        linkstate_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        executor: Executor,
    ) -> None:
        self.linkstate_factory = linkstate_factory
        self.ffs_factory = ffs_factory
        self.executor = executor
        self.executor.initialize(linkstate_factory, ffs_factory)

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")

        run_id = self.executor.start_run(
            request.fab.content,
            user_config_from_proto(request.override_config),
            configs_record_from_proto(request.federation_options),
        )

        if run_id is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        return StartRunResponse(run_id=run_id)

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.StreamLogs")
        state = self.linkstate_factory.state()

        # Retrieve run ID
        run_id = request.run_id

        # Exit if `run_id` not found
        if not state.get_run(run_id):
            context.abort(grpc.StatusCode.NOT_FOUND, "Run ID not found")

        after_timestamp = request.after_timestamp + 1e-6
        while context.is_active():
            log_msg, latest_timestamp = state.get_serverapp_log(run_id, after_timestamp)
            if log_msg:
                yield StreamLogsResponse(
                    log_output=log_msg,
                    latest_timestamp=latest_timestamp,
                )
                # Add a small epsilon to the latest timestamp to avoid getting
                # the same log
                after_timestamp = max(latest_timestamp + 1e-6, after_timestamp)

            # Wait for and continue to yield more log responses only if the
            # run isn't completed yet. If the run is finished, the entire log
            # is returned at this point and the server ends the stream.
            run_status = state.get_run_status({run_id})[run_id]
            if run_status.status == Status.FINISHED:
                log(INFO, "All logs for run ID `%s` returned", request.run_id)
                context.cancel()

            time.sleep(LOG_STREAM_INTERVAL)  # Sleep briefly to avoid busy waiting

    def List(self, request: ListRequest, context: grpc.ServicerContext) -> ListResponse:
        """Handle `flwr ls` command."""
        log(INFO, "ExecServicer.List")
        state = self.linkstate_factory.state()

        # Handle `flwr ls --runs`
        if request.option == "--runs":
            run_ids = state.get_run_ids()
            return _list_runs(run_ids, state)
        # Handle `flwr ls --run-id <run_id>`
        if request.option == "--run-id":
            run_id = scalar_from_proto(request.value)
            if not isinstance(run_id, int):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid run ID")
                return ListResponse()
            return _list_runs({run_id}, state)

        # Unknown option
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Invalid option")
        return ListResponse()


def _list_runs(run_ids: set[int], state: LinkState) -> ListResponse:
    """Create response for `flwr ls --runs` and `flwr ls --run-id <run_id>`."""
    run_status_dict = state.get_run_status(run_ids)
    run_info_dict: dict[int, ListResponse.RunInfo] = {}
    for run_id, run_status in run_status_dict.items():
        run = state.get_run(run_id)
        # Very unlikely, as we just retrieved the run status
        if not run:
            continue
        timestamps = state.get_run_timestamps(run_id)
        run_info_dict[run_id] = ListResponse.RunInfo(
            run=run_to_proto(run),
            status=run_status_to_proto(run_status),
            pending_at=timestamps[0],
            starting_at=timestamps[1],
            running_at=timestamps[2],
            finished_at=timestamps[3],
            now=now().isoformat(),
        )

    return ListResponse(run_info_dict=run_info_dict)

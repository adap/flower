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
from logging import DEBUG, ERROR, INFO
from typing import Any, cast

import grpc

from flwr.common import Context, RecordSet
from flwr.common.config import get_fused_config_from_fab
from flwr.common.constant import LOG_STREAM_INTERVAL, Status
from flwr.common.logger import log
from flwr.common.serde import user_config_from_proto
from flwr.common.typing import Run
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkStateFactory

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
            user_config_from_proto(request.federation_config),
        )

        if run_id is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        # Create a context for the `run_id`
        self._create_context(run_id)

        state = self.linkstate_factory.state()
        run = state.get_run(run_id)
        if run is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"Cannot find the Run with ID: {run_id}"
            )

        # Fuse overrides config from the request to `run_config`
        run_config = get_fused_config_from_fab(request.fab.content, run=cast(Run, run))

        # Update `run_config` in context
        serverapp_context = state.get_serverapp_context(run_id)
        if serverapp_context is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"Cannot find the Context with ID: {run_id}"
            )

        serverapp_context = cast(Context, serverapp_context)
        serverapp_context.run_config = run_config
        state.set_serverapp_context(run_id, serverapp_context)

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

    def _create_context(self, run_id: int) -> None:
        """Register a Context for a Run."""
        log(DEBUG, "ExecServicer._create_context")
        # Create an empty context for the Run
        context = Context(node_id=0, node_config={}, state=RecordSet(), run_config={})

        # Register the context at the LinkState
        state = self.linkstate_factory.state()
        state.set_serverapp_context(run_id=run_id, context=context)

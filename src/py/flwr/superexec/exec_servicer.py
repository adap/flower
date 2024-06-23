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


from logging import ERROR, INFO
from typing import Any, Generator

import grpc

from flwr.common.logger import log
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)

from .executor import Executor
from .state import RunStatus
from .state_factory import SuperexecStateFactory


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """SuperExec API servicer."""

    def __init__(
        self, executor: Executor, state_factory: SuperexecStateFactory
    ) -> None:
        self.executor = executor
        self.state = state_factory.state()

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")

        run = self.executor.start_run(request.fab_file)

        if run is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        self.state.update_run_tracker(run.run_id, RunStatus.RUNNING)

        return StartRunResponse(run_id=run.run_id)

    def StreamLogs(
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        logs = ["a", "b", "c"]
        while context.is_active():
            for i in range(len(logs)):  # pylint: disable=C0200
                yield StreamLogsResponse(log_output=logs[i])

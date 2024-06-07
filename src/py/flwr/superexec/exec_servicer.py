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


from logging import INFO
from subprocess import Popen
from typing import Dict, Generator, Any

import grpc

from flwr.common.logger import log
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    FetchLogsRequest,
    FetchLogsResponse,
)

from .executor import Executor


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """Driver API servicer."""

    def __init__(self, plugin: Executor) -> None:
        self.plugin = plugin
        self.runs: Dict[int, Popen] = {}  # type: ignore

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")
        run = self.plugin.start_run(request.fab_file)
        self.runs[run.run_id] = run.proc
        return StartRunResponse(run_id=run.run_id)

    def FetchLogs(
        self, request: FetchLogsRequest, context: grpc.ServicerContext
    ) -> Generator[FetchLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.FetchLogs")
        proc = self.runs[request.run_id]

        try:
            for line in iter(proc.stderr.readline, ""):
                if context.is_active():
                    yield FetchLogsResponse(log_output=line.strip())
                else:
                    break
        finally:
            proc.stderr.close()
            proc.kill()

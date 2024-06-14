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


import select
import threading
import time
from logging import ERROR, INFO
from typing import Any, Dict, Generator, List

import grpc

from flwr.common.logger import log
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    FetchLogsRequest,
    FetchLogsResponse,
    StartRunRequest,
    StartRunResponse,
)

from .executor import Executor, RunTracker


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """SuperExec API servicer."""

    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        self.runs: Dict[int, RunTracker] = {}
        self.logs: List[str] = []
        self.lock = threading.Lock()

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")
        run = self.executor.start_run(request.fab_file)

        if run is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        self.runs[run.run_id] = run

        # Start background thread to capture logs
        self._capture_logs(run.proc)
        
        return StartRunResponse(run_id=run.run_id)

    def _capture_logs(self, proc: Popen) -> None:  # type: ignore
        select_timeout = 1.0

        def run() -> None:
            while True:
                reads, _, _ = select.select([proc.stderr], [], [], select_timeout)
                if reads and proc.stderr:
                    line = proc.stderr.readline()
                    if line:
                        self.logs.append(line.rstrip())

        threading.Thread(target=run, daemon=True).start()

    def FetchLogs(
        self, request: FetchLogsRequest, context: grpc.ServicerContext
    ) -> Generator[FetchLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.FetchLogs")

        last_sent_index = 0
        while context.is_active():
            with self.lock:
                if last_sent_index < len(self.logs):
                    for i in range(last_sent_index, len(self.logs)):
                        yield FetchLogsResponse(log_output=self.logs[i])
                    last_sent_index = len(self.logs)
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting

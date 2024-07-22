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
from typing import Any, Dict, Generator

import grpc

from flwr.common.logger import log
from flwr.proto import exec_pb2_grpc  # pylint: disable=E0611
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)

from .executor import Executor, RunTracker

SELECT_TIMEOUT = 1  # Timeout for selecting ready-to-read file descriptors (in seconds)


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """SuperExec API servicer."""

    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        self.runs: Dict[int, RunTracker] = {}

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")

        run = self.executor.start_run(
            request.fab_file,
            dict(request.override_config.items()),
        )

        if run is None:
            log(ERROR, "Executor failed to start run")
            return StartRunResponse()

        self.runs[run.run_id] = run

        # Start a background thread to capture the log output
        capture_thread = threading.Thread(
            target=_capture_logs, args=(run,), daemon=True
        )
        capture_thread.start()

        return StartRunResponse(run_id=run.run_id)

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.StreamLogs")

        last_sent_index = 0
        while context.is_active():
            # Exit if `run_id` not found
            if request.run_id not in self.runs:
                context.abort(grpc.StatusCode.NOT_FOUND, "Run ID not found")

            # Yield n'th row of logs, if n'th row < len(logs)
            logs = self.runs[request.run_id].logs
            for i in range(last_sent_index, len(logs)):
                yield StreamLogsResponse(log_output=logs[i])
            last_sent_index = len(logs)

            # Shutdown context if process has completed. Previously stored
            # logs will still be printed.
            if self.runs[request.run_id].proc.poll() is not None:
                log(INFO, "Run ID `%s` completed", request.run_id)
                context.cancel()

            time.sleep(1.0)  # Sleep briefly to avoid busy waiting


def _capture_logs(
    run: RunTracker,
) -> None:
    while not run.stop_event.is_set():
        # Select streams only when ready to read
        ready_to_read, _, _ = select.select(
            [run.proc.stdout, run.proc.stderr],
            [],
            [],
            SELECT_TIMEOUT,
        )
        # Read from std* and append to RunTracker.logs
        for stream in ready_to_read:
            line = stream.readline().rstrip()
            if line:
                run.logs.append(f"{line}")

        # Close std* to prevent blocking
        if run.proc.poll() is not None:
            log(INFO, "Subprocess finished, exiting log capture")
            if run.proc.stdout:
                run.proc.stdout.close()
            if run.proc.stderr:
                run.proc.stderr.close()
            run.stop_event.set()
            break

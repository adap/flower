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
from logging import INFO
from subprocess import Popen
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

from .executor import Executor


class ExecServicer(exec_pb2_grpc.ExecServicer):
    """Driver API servicer."""

    def __init__(self, plugin: Executor) -> None:
        self.plugin = plugin
        self.runs: Dict[int, Popen[str]] = {}

        self.lock = threading.Lock()

        self.select_timeout: int = 1
        self.log_streams = {}

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ExecServicer.StartRun")
        run = self.plugin.start_run(request.fab_file)
        self.runs[run.run_id] = run.proc

        stop_event = threading.Event()
        logs = []
        # Start a background thread to capture the log output
        capture_thread = threading.Thread(
            target=self._capture_logs, args=(run, stop_event, logs), daemon=True
        )
        with self.lock:
            self.log_streams[run.run_id] = {
                "process": run.proc,
                "stop_event": stop_event,
                "logs": logs,
                "capture_thread": capture_thread,
            }
        capture_thread.start()

        return StartRunResponse(run_id=run.run_id)

    def _capture_logs(self, run, stop_event, logs):
        while not stop_event.is_set():
            ready_to_read, _, _ = select.select(
                [run.proc.stdout, run.proc.stderr],
                [],
                [],
                self.select_timeout,
            )
            for stream in ready_to_read:
                line = stream.readline().rstrip()
                if line:
                    with self.lock:
                        logs.append(f"{line}")

            if run.proc.poll() is not None:
                log(INFO, "Subprocess finished, exiting log capture")
                run.proc.stdout.close()
                run.proc.stderr.close()
                stop_event.set()
                break

    def StreamLogs(
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ExecServicer.StreamLogs")

        last_sent_index = 0
        while context.is_active():
            with self.lock:
                if request.run_id not in self.log_streams:
                    context.abort(grpc.StatusCode.NOT_FOUND, "Run ID not found")
                logs = self.log_streams[request.run_id]["logs"]
                if last_sent_index < len(logs):
                    for i in range(last_sent_index, len(logs)):
                        yield StreamLogsResponse(log_output=logs[i])
                    last_sent_index = len(logs)
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting

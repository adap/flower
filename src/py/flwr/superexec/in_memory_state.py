# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""SQLite based implemenation of server state."""


from typing import Dict, List, Optional

from typing_extensions import override

from .state import RunStatus, SuperexecState


class InMemorySuperexecState(SuperexecState):
    """InMemory implementation of SuperexecState."""

    @override
    def initialize(self):
        self.runs: Dict[int, RunStatus] = {}
        self.logs: Dict[int, List[str]] = {}

    @override
    def store_log(self, run_id: int, log_output: str, stream: str = "stderr") -> None:
        """Store logs into the database."""
        if self.logs[run_id]:
            self.logs[run_id].append(log_output)
        else:
            self.logs[run_id] = [log_output]

    @override
    def get_logs(self, run_id: int) -> List[str]:
        """Get logs from the database."""
        return self.logs[run_id]

    @override
    def update_run_tracker(self, run_id: int, status: RunStatus) -> None:
        """Store or update a RunTracker in the database."""
        self.runs[run_id] = status

    @override
    def get_run_tracker_status(self, run_id: int) -> Optional[RunStatus]:
        """Get a RunTracker's status from the database."""
        return self.runs.get(run_id, None)

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
"""SQLite based implemenation of server state."""


from typing import Optional

from typing_extensions import override

from .state import ExecState, RunStatus


class InMemoryExecState(ExecState):
    """InMemory implementation of SuperexecState."""

    def __init__(self) -> None:
        self.runs: dict[int, RunStatus] = {}

    @override
    def update_run_status(self, run_id: int, status: RunStatus) -> None:
        """Store or update a RunStatus in memory."""
        self.runs[run_id] = status

    @override
    def get_run_status(self, run_id: int) -> Optional[RunStatus]:
        """Get a RunStatus from memory."""
        return self.runs.get(run_id, None)

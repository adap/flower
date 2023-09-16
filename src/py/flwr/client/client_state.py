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
"""Client state."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class WorkloadState:
    """Client state when performing a particular workload."""

    cid: Optional[str]
    workload_id: str

    def __repr__(self) -> str:
        """Return a string representation of a ClientState."""
        return (
            f"{self.__class__.__name__}"
            f"(cid:{self.cid}, workload: {self.workload_id}): {self.__dict__}"
        )


@dataclass
class ClientState:
    """Client state.

    A dataclass that keeps track of a separate state for each workload the client
    executes.
    """

    cid: Optional[str] = None
    workload_states: Dict[str, WorkloadState] = field(default_factory=dict)

    def register_workload(self, workload_id: str) -> None:
        """Register a new WorkloadState for a client if it does not exist."""
        if workload_id not in self.workload_states:
            self.workload_states[workload_id] = WorkloadState(self.cid, workload_id)

    def __getitem__(self, workload_id: str) -> WorkloadState:
        """Get client's state for a particular workload."""
        return self.workload_states[workload_id]

    def update_workload_state(self, workload_state: Optional[WorkloadState]) -> None:
        """Update workload state with the one passed."""
        if workload_state:
            self.workload_states[workload_state.workload_id] = workload_state

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
"""Node state."""


from typing import Any, Dict

from flwr.client.run_state import RunState


class NodeState:
    """State of a node where client nodes execute runs."""

    def __init__(self) -> None:
        self._meta: Dict[str, Any] = {}  # holds metadata about the node
        self.run_states: Dict[int, RunState] = {}

    def register_runstate(self, run_id: int) -> None:
        """Register new run state for this node."""
        if run_id not in self.run_states:
            self.run_states[run_id] = RunState({})

    def retrieve_runstate(self, run_id: int) -> RunState:
        """Get run state given a run_id."""
        if run_id in self.run_states:
            return self.run_states[run_id]

        raise RuntimeError(
            f"RunState for run_id={run_id} doesn't exist."
            " A run must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_runstate(self, run_id: int, run_state: RunState) -> None:
        """Update run state."""
        self.run_states[run_id] = run_state

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


from pathlib import Path
from typing import Any, Dict, Optional

from flwr.common import Context, RecordSet
from flwr.common.config import get_fused_config
from flwr.common.typing import Run


class NodeState:
    """State of a node where client nodes execute runs."""

    def __init__(self, partition_id: Optional[int]) -> None:
        self._meta: Dict[str, Any] = {}  # holds metadata about the node
        self.run_contexts: Dict[int, Context] = {}
        self._initial_run_configs: Dict[int, Dict[str, str]] = {}
        self._partition_id = partition_id

    def register_context(
        self,
        run_id: int,
        run_info: Optional[Run] = None,
        flwr_dir: Optional[Path] = None,
    ) -> None:
        """Register new run context for this node."""
        if run_id not in self.run_contexts:
            self._initial_run_configs[run_id] = (
                get_fused_config(run_info, flwr_dir) if run_info else {}
            )
            self.run_contexts[run_id] = Context(
                state=RecordSet(),
                run_config=self._initial_run_configs[run_id].copy(),
                partition_id=self._partition_id,
            )

    def retrieve_context(self, run_id: int) -> Context:
        """Get run context given a run_id."""
        if run_id in self.run_contexts:
            return self.run_contexts[run_id]

        raise RuntimeError(
            f"Context for run_id={run_id} doesn't exist."
            " A run context must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_context(self, run_id: int, context: Context) -> None:
        """Update run context."""
        if context.run_config != self._initial_run_configs[run_id]:
            raise ValueError(
                "The `run_config` field of the `Context` object cannot be "
                f"modified (run_id: {run_id})."
            )
        self.run_contexts[run_id] = context

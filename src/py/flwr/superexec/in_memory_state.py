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

from .state import ExecState
from flwr.common.typing import UserConfig


class InMemoryExecState(ExecState):
    """InMemory implementation of ExecState."""

    def __init__(self) -> None:
        self.runs: dict[int, tuple[UserConfig, str]] = {}

    @override
    def store_run(self, run_id: int, run_config: UserConfig, fab_hash: str) -> None:
        self.runs[run_id] = (run_config, fab_hash)

    @override
    def get_run_config(self, run_id: int) -> Optional[UserConfig]:
        run = self.runs.get(run_id)
        if run:
            return run[0]

    @override
    def get_fab_hash(self, run_id: int) -> Optional[str]:
        run = self.runs.get(run_id)
        if run:
            return run[1]

    @override
    def get_runs(self) -> list[int]:
        return list(self.runs.keys())

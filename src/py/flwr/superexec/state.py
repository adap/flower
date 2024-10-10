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
"""SuperExec state abstraction."""

from abc import ABC, abstractmethod
from typing import Optional

from flwr.common.typing import UserConfig


class ExecState(ABC):
    """Abstract ExecState."""

    @abstractmethod
    def store_run(self, run_id: int, run_config: UserConfig, fab_hash: str) -> None:
        """Store a Run with the given run_id, config, and FAB hash."""

    @abstractmethod
    def get_run_config(self, run_id: int) -> Optional[UserConfig]:
        """Retrieve the run_config of a Run by run_id."""

    @abstractmethod
    def get_fab_hash(self, run_id: int) -> Optional[str]:
        """Retrieve the hash of a FAB from a Run by run_id."""

    @abstractmethod
    def get_runs(self) -> list[int]:
        """Retrieve all the stored run_ids."""

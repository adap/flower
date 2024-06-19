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
"""Factory class that creates State instances."""


from logging import DEBUG
from typing import Optional

from flwr.common.logger import log

from .in_memory_state import InMemoryState
from .sqlite_state import SqliteState
from .state import State


class StateFactory:
    """Factory class that creates State instances."""

    def __init__(self, database: str) -> None:
        self.database = database
        self.state_instance: Optional[State] = None

    def state(self) -> State:
        """Return a State instance and create it, if necessary."""
        # InMemoryState
        if self.database == ":flwr-in-memory-state:":
            if self.state_instance is None:
                self.state_instance = InMemoryState()
            log(DEBUG, "Using InMemoryState")
            return self.state_instance

        # SqliteState
        state = SqliteState(self.database)
        state.initialize()
        log(DEBUG, "Using SqliteState")
        return state

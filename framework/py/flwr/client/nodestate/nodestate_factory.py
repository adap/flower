# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Factory class that creates NodeState instances."""


import threading
from logging import DEBUG
from typing import Optional

from flwr.common.logger import log

from .in_memory_nodestate import InMemoryNodeState
from .nodestate import NodeState
from .sqlite_nodestate import SqliteNodeState


class NodeStateFactory:
    """Factory class that creates NodeState instances.

    Parameters
    ----------
    database : str (default=":flwr-in-memory-state:")
        A string representing the path to the database file that will be opened.
        Note that passing ':memory:' will open a connection to a database that is
        in RAM, instead of on disk. For more information on special in-memory
        databases, please refer to https://sqlite.org/inmemorydb.html.
    """

    def __init__(self, database: str = ":flwr-in-memory-state:") -> None:
        self.database = database
        self.state_instance: Optional[NodeState] = None
        self.lock = threading.RLock()

    def state(self) -> NodeState:
        """Return a State instance and create it, if necessary."""
        # Lock access to NodeStateFactory to prevent returning different instances
        with self.lock:
            # InMemoryNodeState
            if self.database == ":flwr-in-memory-state:":
                if self.state_instance is None:
                    self.state_instance = InMemoryNodeState()
                log(DEBUG, "Using InMemoryNodeState")
                return self.state_instance

            # SqliteNodeState
            state = SqliteNodeState(self.database)
            state.initialize()
            log(DEBUG, "Using SqliteNodeState")
            return state

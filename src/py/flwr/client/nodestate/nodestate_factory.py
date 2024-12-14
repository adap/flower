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
"""Factory class that creates NodeState instances."""


import threading
from typing import Optional

from .in_memory_nodestate import InMemoryNodeState
from .nodestate import NodeState


class NodeStateFactory:
    """Factory class that creates NodeState instances."""

    def __init__(self) -> None:
        self.state_instance: Optional[NodeState] = None
        self.lock = threading.RLock()

    def state(self) -> NodeState:
        """Return a State instance and create it, if necessary."""
        # Lock access to NodeStateFactory to prevent returning different instances
        with self.lock:
            if self.state_instance is None:
                self.state_instance = InMemoryNodeState()
            return self.state_instance

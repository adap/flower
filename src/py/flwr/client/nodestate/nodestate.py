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
"""Abstract base class NodeState."""


import abc
from typing import Optional


class NodeState(abc.ABC):
    """Abstract NodeState."""

    @abc.abstractmethod
    def set_node_id(self, node_id: Optional[int]) -> None:
        """Set the node ID."""

    @abc.abstractmethod
    def get_node_id(self) -> int:
        """Get the node ID."""

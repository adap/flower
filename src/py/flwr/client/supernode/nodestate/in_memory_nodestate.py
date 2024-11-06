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
"""In-memory NodeState implementation."""


from typing import cast

from flwr.client.supernode.nodestate.nodestate import NodeState


class InMemoryNodeState(NodeState):
    """In-memory NodeState implementation."""

    def __init__(self) -> None:

        # Map run_id to node_id
        self.node_id: dict[int, int] = {}

    def set_node_id(self, run_id: int, node_id: int) -> None:
        """Set the node ID for a specified run ID."""
        self.node_id[run_id] = node_id

    def get_node_id(self, run_id: int) -> int:
        """Get the node ID for a specified run ID."""
        return cast(int, self.node_id.get(run_id))

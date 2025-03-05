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
"""Tests all NodeState implementations have to conform to."""


import unittest
from abc import abstractmethod

from flwr.client.nodestate import InMemoryNodeState, NodeState


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def state_factory(self) -> NodeState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_get_set_node_id(self) -> None:
        """Test set_node_id."""
        # Prepare
        state: NodeState = self.state_factory()
        node_id = 123

        # Execute
        state.set_node_id(node_id)

        retrieved_node_id = state.get_node_id()

        # Assert
        assert node_id == retrieved_node_id

    def test_get_node_id_fails(self) -> None:
        """Test get_node_id fails correctly if node_id is not set."""
        # Prepare
        state: NodeState = self.state_factory()

        # Execute and assert
        with self.assertRaises(ValueError):
            state.get_node_id()


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> NodeState:
        """Return InMemoryState."""
        return InMemoryNodeState()


if __name__ == "__main__":
    unittest.main(verbosity=2)

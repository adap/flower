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

from flwr.client.clientapp.clientappio_servicer import ClientAppInputs
from flwr.client.supernode.nodestate import InMemoryNodeState, NodeState
from flwr.common import Context, Message, typing
from flwr.common.serde_test import RecordMaker


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    def setUp(self) -> None:
        """Initialize."""
        self.maker = RecordMaker()

    @abstractmethod
    def state_factory(self) -> NodeState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_get_set_clientapp_inputs(self) -> None:
        """Test set_clientapp_inputs."""
        # Prepare
        state: NodeState = self.state_factory()
        message = Message(
            metadata=self.maker.metadata(),
            content=self.maker.recordset(2, 2, 1),
        )
        context = Context(
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recordset(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        run = typing.Run(
            run_id=1,
            fab_id="lorem",
            fab_version="ipsum",
            fab_hash="dolor",
            override_config=self.maker.user_config(),
        )
        fab = typing.Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
        )
        inputs = ClientAppInputs(message, context, run, fab, 1)
        token = 123

        # Execute
        state.set_clientapp_inputs(token, inputs)
        retrieved_inputs = state.get_clientapp_inputs(token)

        # Assert
        assert retrieved_inputs is not None
        self.assertEqual(inputs, retrieved_inputs)


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> NodeState:
        """Return InMemoryState."""
        return InMemoryNodeState()


if __name__ == "__main__":
    unittest.main(verbosity=2)

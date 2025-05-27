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
"""Tests all NodeState implementations have to conform to."""


import unittest
from typing import Any

from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Message, Metadata, RecordDict
from flwr.common.message import make_message
from flwr.common.typing import Run

from . import InMemoryNodeState, NodeState


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    def setUp(self) -> None:
        """Set up the test case."""
        self.state: NodeState = self.state_factory()

    def state_factory(self) -> NodeState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_get_set_node_id(self) -> None:
        """Test set_node_id."""
        # Prepare
        node_id = 123

        # Execute
        self.state.set_node_id(node_id)

        retrieved_node_id = self.state.get_node_id()

        # Assert
        assert node_id == retrieved_node_id

    def test_get_node_id_fails(self) -> None:
        """Test get_node_id fails correctly if node_id is not set."""
        # Execute and assert
        with self.assertRaises(ValueError):
            self.state.get_node_id()

    def test_store_and_get_run(self) -> None:
        """Test storing and retrieving a run."""
        # Prepare
        run = Run.create_empty(61016)
        self.state.store_run(run)

        # Execute
        retrieved = self.state.get_run(61016)

        # Assert
        self.assertEqual(retrieved, run)

    def test_get_run_ids(self) -> None:
        """Test retrieving run IDs."""
        # Prepare
        run1 = Run.create_empty(61017)
        run2 = Run.create_empty(61018)
        self.state.store_run(run1)
        self.state.store_run(run2)

        # Execute
        ids = self.state.get_run_ids_with_pending_messages()

        # Assert
        self.assertCountEqual(set(ids), {61017, 61018})

    def test_store_and_get_context(self) -> None:
        """Test storing and retrieving a context."""
        # Prepare
        ctx = Context(
            run_id=99,
            node_id=1,
            node_config={"key1": "value1"},
            state=RecordDict({"cfg": ConfigRecord({"key2": "value2"})}),
            run_config={"key3": "value3"},
        )
        self.state.store_context(ctx)

        # Execute
        retrieved = self.state.get_context(99)

        # Assert
        self.assertEqual(retrieved, ctx)

    def test_store_and_get_message_basic(self) -> None:
        """Test storing and retrieving a message."""
        # Prepare
        msg = make_dummy_message()
        obj_id = "abc"

        # Execute
        self.state.store_message(msg, obj_id)

        # Basic retrieval with no filters
        result = self.state.get_message()
        self.assertIn(obj_id, result)
        self.assertEqual(result[obj_id], msg)

        # Ensure message was deleted after retrieval
        result_after = self.state.get_message()
        self.assertNotIn(obj_id, result_after)

    @parameterized.expand(  # type: ignore
        [
            ({"run_id": 1}, {"msg1", "msg2"}),
            ({"run_id": 1, "is_reply": False}, {"msg2"}),
            ({"run_id": 1, "limit": 1}, {"msg1", "msg2"}),
            ({"run_id": [2, 3]}, {"msg3", "msg4"}),
            ({"is_reply": True}, {"msg1", "msg4"}),
            ({"is_reply": True, "limit": 1}, {"msg1", "msg4"}),
        ]
    )
    def test_get_message_with_filters(
        self, filters: dict[str, Any], expected: set[str]
    ) -> None:
        """Test retrieving messages with various filters."""
        # Prepare
        # Run 1: 1 instruction, 1 reply
        self.state.store_message(make_dummy_message(run_id=1, is_reply=True), "msg1")
        self.state.store_message(make_dummy_message(run_id=1, is_reply=False), "msg2")
        # Run 2: 1 instruction
        self.state.store_message(make_dummy_message(run_id=2, is_reply=False), "msg3")
        # Run 3: 1 reply
        self.state.store_message(make_dummy_message(run_id=3, is_reply=True), "msg4")

        # Execute
        result = self.state.get_message(**filters)
        result_ids = set(result.keys())

        # Assert
        if (limit := filters.get("limit")) is not None:
            self.assertEqual(len(result), limit)
            self.assertTrue(result_ids.issubset(expected))
        else:
            self.assertEqual(result_ids, expected)


def make_dummy_message(run_id: int = 110, is_reply: bool = False) -> Message:
    """Create a dummy message for testing."""
    metadata = Metadata(
        run_id=run_id,
        message_id="",
        src_node_id=0,
        dst_node_id=120,
        reply_to_message_id="mock id" if is_reply else "",
        group_id="Mock mock",
        created_at=123456789,
        ttl=999,
        message_type="query",
    )
    content = RecordDict({"cfg": ConfigRecord({"key": "value"})})
    return make_message(metadata, content)


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> NodeState:
        """Return InMemoryState."""
        return InMemoryNodeState()

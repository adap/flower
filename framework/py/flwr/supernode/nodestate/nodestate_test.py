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


from typing import Any

from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Message, Metadata, RecordDict
from flwr.common.message import make_message
from flwr.common.typing import Run
from flwr.supercore.corestate.corestate_test import StateTest as CoreStateTest

from . import InMemoryNodeState, NodeState


class StateTest(CoreStateTest):
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
        msg = make_dummy_message(msg_id="test_msg")

        # Execute
        self.state.store_message(msg)

        # Basic retrieval with no filters
        retrieved_msg = self.state.get_messages()[0]

        self.assertIn("test_msg", retrieved_msg.metadata.message_id)
        self.assertEqual(retrieved_msg, msg)

        # Ensure message won't be retrieved again
        result = self.state.get_messages()
        self.assertEqual(len(result), 0)

    @parameterized.expand(  # type: ignore
        [
            ({"run_ids": [1]}, {"msg1", "msg2"}),
            ({"run_ids": [1], "is_reply": False}, {"msg2"}),
            ({"run_ids": [1], "limit": 1}, {"msg1", "msg2"}),
            ({"run_ids": [2, 3]}, {"msg3", "msg4"}),
            ({"is_reply": True}, {"msg1", "msg4"}),
            ({"is_reply": True, "limit": 1}, {"msg1", "msg4"}),
            ({"is_retrieved": False}, {"msg1", "msg2", "msg3", "msg4"}),
            ({"is_retrieved": True}, set()),
            ({"run_ids": [1], "is_retrieved": False}, {"msg1", "msg2"}),
        ]
    )
    def test_get_message_with_filters(
        self, filters: dict[str, Any], expected: set[str]
    ) -> None:
        """Test retrieving messages with various filters."""
        # Prepare
        # Run 1: 1 instruction, 1 reply
        self.state.store_message(make_dummy_message(1, True, "msg1"))
        self.state.store_message(make_dummy_message(1, False, "msg2"))
        # Run 2: 1 instruction
        self.state.store_message(make_dummy_message(2, False, "msg3"))
        # Run 3: 1 reply
        self.state.store_message(make_dummy_message(3, True, "msg4"))

        # Execute
        result = self.state.get_messages(**filters)
        result_ids = {msg.metadata.message_id for msg in result}

        # Assert
        if (limit := filters.get("limit")) is not None:
            self.assertEqual(len(result), limit)
            self.assertTrue(result_ids.issubset(expected))
        else:
            self.assertEqual(result_ids, expected)

    def test_delete_message(self) -> None:
        """Test deleting messages."""
        # Prepare
        msg1 = make_dummy_message(msg_id="msg1")
        msg2 = make_dummy_message(msg_id="msg2")
        self.state.store_message(msg1)
        self.state.store_message(msg2)

        # Execute: delete one message
        self.state.delete_messages(message_ids=["msg1"])

        # Assert: msg1 should be deleted, msg2 should remain
        msgs = self.state.get_messages()
        msg_ids = {msg.metadata.message_id for msg in msgs}
        self.assertNotIn("msg1", msg_ids)
        self.assertIn("msg2", msg_ids)

    def test_get_messages_is_retrieved_filter(self) -> None:
        """Test retrieving messages based on is_retrieved status."""
        # Prepare: store messages
        self.state.store_message(make_dummy_message(1, False, "msg1"))
        self.state.store_message(make_dummy_message(2, False, "msg2"))
        self.state.store_message(make_dummy_message(3, False, "msg3"))

        # Execute: retrieve msg1 and msg2
        result1 = self.state.get_messages(run_ids=[1, 2], is_retrieved=False)
        self.assertEqual(len(result1), 2)

        # Assert: can retrieve already retrieved messages with is_retrieved=True
        result2 = self.state.get_messages(is_retrieved=True)
        result2_ids = {msg.metadata.message_id for msg in result2}
        self.assertEqual(result2_ids, {"msg1", "msg2"})

        # Assert: msg3 is still not retrieved
        result3 = self.state.get_messages(is_retrieved=False)
        result3_ids = {msg.metadata.message_id for msg in result3}
        self.assertEqual(result3_ids, {"msg3"})

        # Assert: can retrieve all messages with is_retrieved=None
        result4 = self.state.get_messages(is_retrieved=None)
        result4_ids = {msg.metadata.message_id for msg in result4}
        # All three messages should be returned
        self.assertEqual(result4_ids, {"msg1", "msg2", "msg3"})

    def test_get_run_ids_with_pending_messages(self) -> None:
        """Test retrieving run IDs with pending messages."""
        # Prepare: store messages for runs 1, 2, and 3
        # Run 1 has a pending message, run 2 has a token, run 3 has a reply,
        # run 4 has a retrieved message (not pending),
        #  and run 5 was assigned a token but was later deleted due to
        # `flwr-clientapp` finishing the handling of a message.
        self.state.store_message(make_dummy_message(1, False, "msg1"))
        self.state.store_message(make_dummy_message(2, False, "msg2"))
        self.state.store_message(make_dummy_message(3, True, "msg3"))
        self.state.store_message(make_dummy_message(4, False, "msg4"))
        self.state.store_message(make_dummy_message(5, False, "msg5"))
        self.state.get_messages(run_ids=[4])
        self.state.create_token(2)
        self.state.create_token(5)
        self.state.delete_token(5)

        # Execute
        run_ids = self.state.get_run_ids_with_pending_messages()

        # Assert: run 1 and run 5 should be returned
        self.assertEqual(set(run_ids), {1, 5})


def make_dummy_message(
    run_id: int = 110, is_reply: bool = False, msg_id: str = ""
) -> Message:
    """Create a dummy message for testing."""
    metadata = Metadata(
        run_id=run_id,
        # This is for testing purposes, in a real scenario this would be `.object_id`
        message_id=msg_id,
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

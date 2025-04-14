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
"""Tests for in-memory grid."""


import time
import unittest
from collections.abc import Iterable
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from flwr.common import ConfigRecord, Message, RecordDict, now
from flwr.common.constant import (
    NODE_ID_NUM_BYTES,
    PING_MAX_INTERVAL,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.serde import message_from_proto
from flwr.common.typing import Run, RunStatus
from flwr.server.superlink.linkstate import (
    InMemoryLinkState,
    LinkStateFactory,
    SqliteLinkState,
)
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes

from .inmemory_grid import InMemoryGrid


def push_messages(grid: InMemoryGrid, num_nodes: int) -> tuple[Iterable[str], int]:
    """Help push messages to state."""
    for _ in range(num_nodes):
        node_id = grid.state.create_node(ping_interval=PING_MAX_INTERVAL)
    num_messages = 3
    msgs = [Message(RecordDict(), node_id, "query") for _ in range(num_messages)]

    # Execute: push messages
    return grid.push_messages(msgs), node_id


def get_replies(grid: InMemoryGrid, msg_ids: Iterable[str], node_id: int) -> list[str]:
    """Help create message replies and pull them from state."""
    messages = grid.state.get_message_ins(node_id, limit=len(list(msg_ids)))
    for msg in messages:
        reply_msg = Message(RecordDict(), reply_to=msg)
        grid.state.store_message_res(message=reply_msg)

    # Execute: Pull messages
    pulled_msgs = grid.pull_messages(msg_ids)
    return [msg.metadata.reply_to_message_id for msg in pulled_msgs]


class TestInMemoryGrid(unittest.TestCase):
    """Tests for `InMemoryGrid` class."""

    def setUp(self) -> None:
        """Initialize State and Grid instance before each test.

        Grid uses the default StateFactory (i.e. SQLite)
        """
        # Create grid
        self.num_nodes = 42
        self.state = MagicMock()
        self.state.get_nodes.return_value = [
            generate_rand_int_from_bytes(NODE_ID_NUM_BYTES)
            for _ in range(self.num_nodes)
        ]
        self.state.get_run.return_value = Run(
            run_id=61016,
            fab_id="mock/mock",
            fab_version="v1.0.0",
            fab_hash="9f86d08",
            override_config={"test_key": "test_value"},
            pending_at=now().isoformat(),
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(status=Status.PENDING, sub_status="", details=""),
        )
        state_factory = MagicMock(state=lambda: self.state)
        self.grid = InMemoryGrid(state_factory=state_factory)
        self.grid.set_run(run_id=61016)
        self.grid.state = self.state

    def test_get_run(self) -> None:
        """Test the InMemoryGrid starting with run_id."""
        # Assert
        self.assertEqual(self.grid.run.run_id, 61016)
        self.assertEqual(self.grid.run.fab_id, "mock/mock")
        self.assertEqual(self.grid.run.fab_version, "v1.0.0")
        self.assertEqual(self.grid.run.fab_hash, "9f86d08")
        self.assertEqual(self.grid.run.override_config["test_key"], "test_value")

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Execute
        node_ids = list(self.grid.get_node_ids())

        # Assert
        self.assertEqual(len(node_ids), self.num_nodes)

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        num_messages = 2
        msgs = [Message(RecordDict(), 1, "query") for _ in range(num_messages)]

        msg_ids = [uuid4() for _ in range(num_messages)]
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        msg_res_ids = list(self.grid.push_messages(msgs))

        # Assert
        self.assertEqual(len(msg_res_ids), 2)
        self.assertEqual(msg_res_ids, [str(ids) for ids in msg_ids])

    def test_push_messages_invalid(self) -> None:
        """Test pushing invalid messages."""
        # Prepare
        msgs = [Message(RecordDict(), 1, "query") for _ in range(2)]
        # Use invalid run_id
        msgs[1].metadata.__dict__["_message_id"] = "invalid message id"

        # Execute and assert
        with self.assertRaises(ValueError):
            self.grid.push_messages(msgs)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list

        # Execute
        pulled_msgs = list(self.grid.pull_messages(msg_ids))
        reply_tos = [msg.metadata.reply_to_message_id for msg in pulled_msgs]

        # Assert
        self.assertEqual(len(pulled_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)
        # Ensure messages are deleted
        self.state.delete_messages.assert_called_once_with(
            message_ins_ids={UUID(m_id) for m_id in msg_ids}
        )

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        msgs = [Message(RecordDict(), 0, "query")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        ret_msgs = list(self.grid.send_and_receive(msgs))
        reply_tos = [msg.metadata.reply_to_message_id for msg in ret_msgs]
        # Assert
        self.assertEqual(len(ret_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        msgs = [Message(RecordDict(), 0, "query")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        with patch("time.sleep", side_effect=lambda t: time.sleep(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.grid.send_and_receive(msgs, timeout=-1))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_message_store_consistency_after_push_pull_sqlitestate(self) -> None:
        """Test messages are deleted in sqlite state once messages are pulled."""
        # Prepare
        state = LinkStateFactory("").state()
        run_id = state.create_run("", "", "", {}, ConfigRecord())
        self.grid = InMemoryGrid(MagicMock(state=lambda: state))
        self.grid.set_run(run_id=run_id)
        msg_ids, node_id = push_messages(self.grid, self.num_nodes)
        assert isinstance(state, SqliteLinkState)

        # Check recorded
        num_msg_ins = len(state.query("SELECT * FROM message_ins;"))
        self.assertEqual(num_msg_ins, len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.grid, msg_ids, node_id)

        # Query number of Messages and reply Messages in State
        num_msg_res = len(state.query("SELECT * FROM message_res;"))
        num_msg_ins = len(state.query("SELECT * FROM message_ins;"))

        # Assert
        self.assertEqual(reply_tos, msg_ids)
        self.assertEqual(num_msg_res, 0)
        self.assertEqual(num_msg_ins, 0)

    def test_message_store_consistency_after_push_pull_inmemory_state(self) -> None:
        """Test messages are deleted in in-memory state once messages are pulled."""
        # Prepare
        state_factory = LinkStateFactory(":flwr-in-memory-state:")
        state = state_factory.state()
        run_id = state.create_run("", "", "", {}, ConfigRecord())
        self.grid = InMemoryGrid(state_factory)
        self.grid.set_run(run_id=run_id)
        msg_ids, node_id = push_messages(self.grid, self.num_nodes)
        assert isinstance(state, InMemoryLinkState)

        # Check recorded
        self.assertEqual(len(state.message_ins_store), len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.grid, msg_ids, node_id)

        # Assert
        self.assertEqual(set(reply_tos), set(msg_ids))
        self.assertEqual(len(state.message_res_store), 0)
        self.assertEqual(len(state.message_ins_store), 0)


def create_message_replies_for_specific_ids(message_ids: list[str]) -> list[Message]:
    """Create reply Messages for a set of message IDs."""
    message_replies = []

    for msg_id in message_ids:

        message = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=123, run_id=456
            )
        )
        # pylint: disable=W0212
        message.metadata._message_id = msg_id  # type: ignore

        # Append reply
        message_replies.append(Message(RecordDict(), reply_to=message))

    return message_replies

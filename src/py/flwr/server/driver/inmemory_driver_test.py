# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for in-memory driver."""


import unittest
import time

from unittest.mock import MagicMock
from flwr.common import DEFAULT_TTL, RecordSet
from flwr.common.constant import PING_MAX_INTERVAL
from flwr.server.superlink.state import StateFactory
from flwr.common.serde import message_to_taskres, message_from_taskins

from .inmemory_driver import InMemoryDriver


class TestInMemoryDriver(unittest.TestCase):
    """Tests for `InMemoryDriver` class."""

    def setUp(self) -> None:
        """Initialize State and Driver instance before each test."""
        state_factory = StateFactory(":flwr-in-memory-state:")

        # Register a few nodes
        self.num_nodes = 42
        state = state_factory.state()
        for _ in range(self.num_nodes):
            state.create_node(ping_interval=PING_MAX_INTERVAL)

        # Create driver
        self.driver = InMemoryDriver(state_factory)

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Execute
        node_ids = self.driver.get_node_ids()

        # Assert
        self.assertEqual(len(node_ids), self.num_nodes)

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        msgs = [
            self.driver.create_message(RecordSet(), "message_type", 1, "")
            for _ in range(2)
        ]

        # Execute
        msg_ids = self.driver.push_messages(msgs)

        # Assert
        self.assertEqual(len(msg_ids), 2)

    def test_push_messages_invalid(self) -> None:
        """Test pushing invalid messages."""
        # Prepare
        msgs = [
            self.driver.create_message(RecordSet(), "message_type", 1, "")
            for _ in range(2)
        ]
        # Use invalid run_id
        msgs[1].metadata._run_id += 1  # pylint: disable=protected-access

        # Execute and assert
        with self.assertRaises(ValueError):
            self.driver.push_messages(msgs)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare: push messages
        num_messages = 3
        node_id = 1
        msgs = [
            self.driver.create_message(RecordSet(), "message_type", node_id, "")
            for _ in range(num_messages)
        ]
        msg_ids = self.driver.push_messages(msgs)
        
        # Prepare: create replies
        taskins = self.driver.state.get_task_ins(node_id, limit = num_messages)
        for taskin in taskins:
            msg = message_from_taskins(taskin)
            reply_msg = msg.create_reply(RecordSet())
            task_res = message_to_taskres(reply_msg)
            task_res.task.pushed_at = time.time()
            self.driver.state.store_task_res(task_res=task_res)

        # Execute
        pulled_msgs = self.driver.pull_messages(msg_ids)
        reply_tos = [msg.metadata.reply_to_message for msg in pulled_msgs]

        # # Assert
        self.assertEqual(len(msgs), num_messages)
        self.assertEqual(reply_tos, msg_ids)

    # def test_send_and_receive_messages_complete(self) -> None:
    #     """Test send and receive all messages successfully."""
    #     # Prepare
    #     mock_response = Mock(task_ids=["id1"])
    #     self.mock_grpc_driver_helper.push_task_ins.return_value = mock_response
    #     # The response message must include either `content` (i.e. a recordset) or
    #     # an `Error`. We choose the latter in this case
    #     error_proto = error_to_proto(Error(code=0))
    #     mock_response = Mock(
    #         task_res_list=[TaskRes(task=Task(ancestry=["id1"], error=error_proto))]
    #     )
    #     self.mock_grpc_driver_helper.pull_task_res.return_value = mock_response
    #     msgs = [self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)]

    #     # Execute
    #     ret_msgs = list(self.driver.send_and_receive(msgs))

    #     # Assert
    #     self.assertEqual(len(ret_msgs), 1)
    #     self.assertEqual(ret_msgs[0].metadata.reply_to_message, "id1")

    # def test_send_and_receive_messages_timeout(self) -> None:
    #     """Test send and receive messages but time out."""
    #     # Prepare
    #     sleep_fn = time.sleep
    #     mock_response = Mock(task_ids=["id1"])
    #     self.mock_grpc_driver_helper.push_task_ins.return_value = mock_response
    #     mock_response = Mock(task_res_list=[])
    #     self.mock_grpc_driver_helper.pull_task_res.return_value = mock_response
    #     msgs = [self.driver.create_message(RecordSet(), "", 0, "", DEFAULT_TTL)]

    #     # Execute
    #     with patch("time.sleep", side_effect=lambda t: sleep_fn(t * 0.01)):
    #         start_time = time.time()
    #         ret_msgs = list(self.driver.send_and_receive(msgs, timeout=0.15))

    #     # Assert
    #     self.assertLess(time.time() - start_time, 0.2)
    #     self.assertEqual(len(ret_msgs), 0)

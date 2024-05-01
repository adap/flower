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


import os
import time
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from flwr.common import RecordSet
from flwr.common.message import Error
from flwr.common.serde import error_to_proto, recordset_to_proto
from flwr.proto.task_pb2 import Task, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.state import StateFactory

from .inmemory_driver import InMemoryDriver


class TestInMemoryDriver(unittest.TestCase):
    """Tests for `InMemoryDriver` class."""

    def setUp(self) -> None:
        """Initialize State and Driver instance before each test."""
        # Create driver
        state_factory = StateFactory(":flwr-in-memory-state:")
        self.driver = InMemoryDriver(state_factory)

        self.num_nodes = 42
        self.driver.state = MagicMock()
        self.driver.state.get_nodes.return_value = [
            int.from_bytes(os.urandom(8), "little", signed=True)
            for _ in range(self.num_nodes)
        ]

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Execute
        node_ids = self.driver.get_node_ids()

        # Assert
        self.assertEqual(len(node_ids), self.num_nodes)

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        num_messages = 2
        msgs = [
            self.driver.create_message(RecordSet(), "message_type", 1, "")
            for _ in range(num_messages)
        ]

        taskins_ids = [uuid4() for _ in range(num_messages)]
        self.driver.state.store_task_ins.side_effect = taskins_ids  # type: ignore

        # Execute
        msg_ids = list(self.driver.push_messages(msgs))

        # Assert
        self.assertEqual(len(msg_ids), 2)
        self.assertEqual(msg_ids, [str(ids) for ids in taskins_ids])

    def test_push_messages_invalid(self) -> None:
        """Test pushing invalid messages."""
        # Prepare
        msgs = [
            self.driver.create_message(RecordSet(), "message_type", 1, "")
            for _ in range(2)
        ]
        # Use invalid run_id
        msgs[1].metadata._run_id += 1  # type: ignore

        # Execute and assert
        with self.assertRaises(ValueError):
            self.driver.push_messages(msgs)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        task_res_list = [
            TaskRes(
                task=Task(
                    ancestry=[msg_ids[0]], recordset=recordset_to_proto(RecordSet())
                )
            ),
            TaskRes(
                task=Task(ancestry=[msg_ids[1]], error=error_to_proto(Error(code=0)))
            ),
        ]
        self.driver.state.get_task_res.return_value = task_res_list  # type: ignore

        # Execute
        pulled_msgs = list(self.driver.pull_messages(msg_ids))
        reply_tos = [msg.metadata.reply_to_message for msg in pulled_msgs]

        # Assert
        self.assertEqual(len(pulled_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        msgs = [self.driver.create_message(RecordSet(), "", 0, "")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        task_res_list = [
            TaskRes(
                task=Task(
                    ancestry=[msg_ids[0]], recordset=recordset_to_proto(RecordSet())
                )
            ),
            TaskRes(
                task=Task(ancestry=[msg_ids[1]], error=error_to_proto(Error(code=0)))
            ),
        ]
        self.driver.state.store_task_ins.side_effect = msg_ids  # type: ignore
        self.driver.state.get_task_res.return_value = task_res_list  # type: ignore

        # Execute
        ret_msgs = list(self.driver.send_and_receive(msgs))
        reply_tos = [msg.metadata.reply_to_message for msg in ret_msgs]
        # Assert
        self.assertEqual(len(ret_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        msgs = [self.driver.create_message(RecordSet(), "", 0, "")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        task_res_list = [
            TaskRes(
                task=Task(
                    ancestry=[msg_ids[0]], recordset=recordset_to_proto(RecordSet())
                )
            ),
            TaskRes(
                task=Task(ancestry=[msg_ids[1]], error=error_to_proto(Error(code=0)))
            ),
        ]
        self.driver.state.store_task_ins.side_effect = msg_ids  # type: ignore
        self.driver.state.get_task_res.return_value = task_res_list  # type: ignore

        # Execute
        with patch("time.sleep", side_effect=lambda t: time.sleep(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.driver.send_and_receive(msgs, timeout=-1))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

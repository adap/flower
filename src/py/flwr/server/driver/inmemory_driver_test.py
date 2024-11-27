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
"""Tests for in-memory driver."""


import time
import unittest
from collections.abc import Iterable
from unittest.mock import MagicMock, patch
from uuid import uuid4

from flwr.common import ConfigsRecord, RecordSet, now
from flwr.common.constant import NODE_ID_NUM_BYTES, PING_MAX_INTERVAL, Status
from flwr.common.message import Error
from flwr.common.serde import (
    error_to_proto,
    message_from_taskins,
    message_to_taskres,
    recordset_to_proto,
)
from flwr.common.typing import Run, RunStatus
from flwr.proto.task_pb2 import Task, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.linkstate import (
    InMemoryLinkState,
    LinkStateFactory,
    SqliteLinkState,
)
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes

from .inmemory_driver import InMemoryDriver


def push_messages(driver: InMemoryDriver, num_nodes: int) -> tuple[Iterable[str], int]:
    """Help push messages to state."""
    for _ in range(num_nodes):
        node_id = driver.state.create_node(ping_interval=PING_MAX_INTERVAL)
    num_messages = 3
    msgs = [
        driver.create_message(RecordSet(), "message_type", node_id, "")
        for _ in range(num_messages)
    ]

    # Execute: push messages
    return driver.push_messages(msgs), node_id


def get_replies(
    driver: InMemoryDriver, msg_ids: Iterable[str], node_id: int
) -> list[str]:
    """Help create message replies and pull taskres from state."""
    taskins = driver.state.get_task_ins(node_id, limit=len(list(msg_ids)))
    for taskin in taskins:
        msg = message_from_taskins(taskin)
        reply_msg = msg.create_reply(RecordSet())
        task_res = message_to_taskres(reply_msg)
        task_res.task.pushed_at = time.time()
        driver.state.store_task_res(task_res=task_res)

    # Execute: Pull messages
    pulled_msgs = driver.pull_messages(msg_ids)
    return [msg.metadata.reply_to_message for msg in pulled_msgs]


class TestInMemoryDriver(unittest.TestCase):
    """Tests for `InMemoryDriver` class."""

    def setUp(self) -> None:
        """Initialize State and Driver instance before each test.

        Driver uses the default StateFactory (i.e. SQLite)
        """
        # Create driver
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
        self.driver = InMemoryDriver(state_factory=state_factory)
        self.driver.set_run(run_id=61016)
        self.driver.state = self.state

    def test_get_run(self) -> None:
        """Test the InMemoryDriver starting with run_id."""
        # Assert
        self.assertEqual(self.driver.run.run_id, 61016)
        self.assertEqual(self.driver.run.fab_id, "mock/mock")
        self.assertEqual(self.driver.run.fab_version, "v1.0.0")
        self.assertEqual(self.driver.run.fab_hash, "9f86d08")
        self.assertEqual(self.driver.run.override_config["test_key"], "test_value")

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
        self.state.store_task_ins.side_effect = taskins_ids

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
        self.state.get_task_res.return_value = task_res_list

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
        self.state.store_task_ins.side_effect = msg_ids
        self.state.get_task_res.return_value = task_res_list

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
        self.state.store_task_ins.side_effect = msg_ids
        self.state.get_task_res.return_value = task_res_list

        # Execute
        with patch("time.sleep", side_effect=lambda t: time.sleep(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.driver.send_and_receive(msgs, timeout=-1))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_task_store_consistency_after_push_pull_sqlitestate(self) -> None:
        """Test tasks are deleted in sqlite state once messages are pulled."""
        # Prepare
        state = LinkStateFactory("").state()
        run_id = state.create_run("", "", "", {}, ConfigsRecord())
        self.driver = InMemoryDriver(MagicMock(state=lambda: state))
        self.driver.set_run(run_id=run_id)
        msg_ids, node_id = push_messages(self.driver, self.num_nodes)
        assert isinstance(state, SqliteLinkState)

        # Check recorded
        task_ins = state.query("SELECT * FROM task_ins;")
        self.assertEqual(len(task_ins), len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.driver, msg_ids, node_id)

        # Query number of task_ins and task_res in State
        task_res = state.query("SELECT * FROM task_res;")
        task_ins = state.query("SELECT * FROM task_ins;")

        # Assert
        self.assertEqual(reply_tos, msg_ids)
        self.assertEqual(len(task_res), 0)
        self.assertEqual(len(task_ins), 0)

    def test_task_store_consistency_after_push_pull_inmemory_state(self) -> None:
        """Test tasks are deleted in in-memory state once messages are pulled."""
        # Prepare
        state_factory = LinkStateFactory(":flwr-in-memory-state:")
        state = state_factory.state()
        run_id = state.create_run("", "", "", {}, ConfigsRecord())
        self.driver = InMemoryDriver(state_factory)
        self.driver.set_run(run_id=run_id)
        msg_ids, node_id = push_messages(self.driver, self.num_nodes)
        assert isinstance(state, InMemoryLinkState)

        # Check recorded
        self.assertEqual(len(state.task_ins_store), len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.driver, msg_ids, node_id)

        # Assert
        self.assertEqual(set(reply_tos), set(msg_ids))
        self.assertEqual(len(state.task_res_store), 0)
        self.assertEqual(len(state.task_ins_store), 0)

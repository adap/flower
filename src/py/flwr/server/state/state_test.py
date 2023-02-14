# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Tests all state implemenations have to conform to."""
# pylint: disable=no-self-use, invalid-name, disable=R0904

import unittest

from .in_memory_state import InMemoryState
from .sqlite_state import SqliteState
from .state import State

from datetime import datetime, timezone
from uuid import uuid4

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns


class StateTest:
    """Test all state implementations."""

    @property
    def state_factory(self) -> State:
        raise NotImplementedError()

    def test_correct_initial_state(self) -> None:
        """Test that new state does not contain any TaskIns."""

        # Prepare
        state = self.state_factory()

        # Execute
        task_ins_list = state.get_task_ins(
            node_id=1,
            limit=10,
        )

        # Assert
        assert type(task_ins_list) == list
        assert not task_ins_list

    def test_get_task_ins_identity(self) -> None:
        """Test get_task_ins retrieves stored task for specific node_id."""

        # Prepare
        state = self.state_factory()
        task_id = state.store_task_ins(
            task_ins=TaskIns(
                task_id="",
                group_id="",
                workload_id="",
                task=Task(
                    producer=Node(node_id=0, anonymous=True),
                    consumer=Node(node_id=123, anonymous=False),
                ),
            )
        )

        # Execute
        task_ins_list = state.get_task_ins(
            node_id=123,
            limit=10,
        )

        # Assert
        assert len(task_ins_list) == 1
        assert task_ins_list[0].task_id == str(task_id)


    def test_get_task_ins_anonymous(self) -> None:
        """Test get_task_ins retrieves stored task for annoynmous node_id."""

        # Prepare
        state = self.state_factory()
        task_id = state.store_task_ins(
            task_ins=TaskIns(
                task_id="",
                group_id="",
                workload_id="",
                task=Task(
                    producer=Node(node_id=0, anonymous=True),
                    consumer=Node(node_id=0, anonymous=True),
                ),
            )
        )

        # Execute
        task_ins_list = state.get_task_ins(
            node_id=None,
            limit=10,
        )

        # Assert
        assert len(task_ins_list) == 1
        assert task_ins_list[0].task_id == str(task_id)


    def test_get_task_res_empty(self) -> None:
        """Test that new state does not contain any TaskRes."""

        # Prepare
        state = self.state_factory()

        # Execute
        task_res_list = state.get_task_res(
            task_ids={uuid4()},
            limit=10,
        )

        # Assert
        assert type(task_res_list) == list
        assert not task_res_list


    def test_store_task_ins_one(self) -> None:
        """Test store_task_ins."""

        # Prepare
        node_id = 1
        state = self.state_factory()
        task_ins: TaskIns = TaskIns(
            task_id=str(uuid4()),
            group_id="",
            workload_id="",
            task=Task(
                consumer=Node(node_id=node_id, anonymous=False),
            ),
        )

        assert task_ins.task.created_at == ""  # pylint: disable=no-member
        assert task_ins.task.delivered_at == ""  # pylint: disable=no-member
        assert task_ins.task.ttl == ""  # pylint: disable=no-member

        # Execute
        state.store_task_ins(task_ins=task_ins)
        task_ins_list = state.get_task_ins(node_id=node_id, limit=10)

        # Assert
        assert len(task_ins_list) == 1

        actual_task_ins = task_ins_list[0]

        assert actual_task_ins.task_id == task_ins.task_id  # pylint: disable=no-member
        assert actual_task_ins.task is not None

        actual_task = actual_task_ins.task

        assert actual_task.created_at != ""
        assert actual_task.delivered_at != ""
        assert actual_task.ttl != ""

        assert datetime.fromisoformat(actual_task.created_at) > datetime(
            2020, 1, 1, tzinfo=timezone.utc
        )
        assert datetime.fromisoformat(actual_task.delivered_at) > datetime(
            2020, 1, 1, tzinfo=timezone.utc
        )
        assert datetime.fromisoformat(actual_task.ttl) > datetime(
            2020, 1, 1, tzinfo=timezone.utc
        )

    # def test_store_task_ins(self) -> None:
    #     """Test storing one TaskIns."""

    # def test_get_task_ins(self) -> None:
    #     """Test getting all TaskIns that have not been delivered yet."""

    # def test_store_task_res(self) -> None:
    #     """Test storing one TaskRes."""

    # def test_get_task_res(self) -> None:
    #     """Test getting all TaskRes that have not been delivered yet."""

    # def test_register_node(self) -> None:
    #     """Test registering a client node."""

    # def test_unregister_node(self) -> None:
    #     """Test unregistering a client node."""

    # def test_get_nodes(self) -> None:
    #     """Test returning all available client nodes."""


class InMemoryStateTest(StateTest, unittest.TestCase):
    state_factory = lambda self: InMemoryState()

class SqliteStateTest(StateTest, unittest.TestCase):
    state_factory = lambda self: SqliteState()

if __name__ == '__main__':
    unittest.main(verbosity=2)

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

import tempfile
import unittest
from abc import abstractmethod
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.state import InMemoryState, SqliteState, State


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def state_factory(self) -> State:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_get_task_ins_empty(self) -> None:
        """Validate that a new state has no TaskIns."""
        # Prepare
        state = self.state_factory()

        # Execute
        num_task_ins = state.num_task_ins()

        # Assert
        assert num_task_ins == 0

    def test_get_task_res_empty(self) -> None:
        """Validate that a new state has no TaskRes."""
        # Prepare
        state = self.state_factory()

        # Execute
        num_tasks_res = state.num_task_res()

        # Assert
        assert num_tasks_res == 0

    def test_store_task_ins_one(self) -> None:
        """Test store_task_ins."""
        # Prepare
        consumer_node_id = 1
        state = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=consumer_node_id, anonymous=False)

        assert task_ins.task.created_at == ""  # pylint: disable=no-member
        assert task_ins.task.delivered_at == ""  # pylint: disable=no-member
        assert task_ins.task.ttl == ""  # pylint: disable=no-member

        # Execute
        state.store_task_ins(task_ins=task_ins)
        task_ins_list = state.get_task_ins(node_id=consumer_node_id, limit=10)

        # Assert
        assert len(task_ins_list) == 1

        actual_task_ins = task_ins_list[0]

        assert actual_task_ins.task_id == task_ins.task_id  # pylint: disable=no-member
        assert actual_task_ins.HasField("task")

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

    def test_store_and_delete_tasks(self) -> None:
        """Test delete_tasks."""
        # Prepare
        consumer_node_id = 1
        state = self.state_factory()
        task_ins_0 = create_task_ins(consumer_node_id=consumer_node_id, anonymous=False)
        task_ins_1 = create_task_ins(consumer_node_id=consumer_node_id, anonymous=False)
        task_ins_2 = create_task_ins(consumer_node_id=consumer_node_id, anonymous=False)

        # Insert three TaskIns
        task_id_0 = state.store_task_ins(task_ins=task_ins_0)
        task_id_1 = state.store_task_ins(task_ins=task_ins_1)
        task_id_2 = state.store_task_ins(task_ins=task_ins_2)

        assert task_id_0
        assert task_id_1
        assert task_id_2

        # Get TaskIns to mark them delivered
        _ = state.get_task_ins(node_id=consumer_node_id, limit=None)

        # Insert one TaskRes and retrive it to mark it as delivered
        task_res_0 = create_task_res(
            producer_node_id=100, anonymous=False, ancestry=[str(task_id_0)]
        )

        _ = state.store_task_res(task_res=task_res_0)
        _ = state.get_task_res(task_ids={task_id_0}, limit=None)

        # Insert one TaskRes, but don't retrive it
        task_res_1: TaskRes = create_task_res(
            producer_node_id=100, anonymous=False, ancestry=[str(task_id_1)]
        )
        _ = state.store_task_res(task_res=task_res_1)

        # Situation now:
        # - State has three TaskIns, all of them delivered
        # - State has two TaskRes, one of the delivered, the other not

        assert state.num_task_ins() == 3
        assert state.num_task_res() == 2

        # Execute
        state.delete_tasks(task_ids={task_id_0, task_id_1, task_id_2})

        # Assert
        assert state.num_task_ins() == 2
        assert state.num_task_res() == 1

    # Init tests
    def test_init_state(self) -> None:
        """Test that state is initialized correctly."""
        # Execute
        state = self.state_factory()

        # Assert
        assert isinstance(state, State)

    # TaskIns tests
    def test_task_ins_store_anonymous_and_retrieve_anonymous(self) -> None:
        """Store one TaskIns.

        Create anonymous task and retrieve it.
        """
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True)

        # Execute
        task_ins_uuid = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=None, limit=None)

        # Assert
        assert len(task_ins_list) == 1
        assert task_ins_list[0].task_id == str(task_ins_uuid)

    def test_task_ins_store_anonymous_and_fail_retrieving_identitiy(self) -> None:
        """Store anonymous TaskIns and fail to retrieve it."""
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True)

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_fail_retrieving_anonymous(self) -> None:
        """Store identity TaskIns and fail retrieving it as anonymous."""
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False)

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=None, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_retrieve_identity(self) -> None:
        """Store identity TaskIns and retrieve it."""
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False)

        # Execute
        task_ins_uuid = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 1

        retrieved_task_ins = task_ins_list[0]
        assert retrieved_task_ins.task_id == str(task_ins_uuid)

    def test_task_ins_store_delivered_and_fail_retrieving(self) -> None:
        """Fail retrieving delivered task."""
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False)

        # Execute
        _ = state.store_task_ins(task_ins)

        # 1st get: set to delivered
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        assert len(task_ins_list) == 1

        # 2nd get: no TaskIns because it was already delivered before
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_get_task_ins_limit_throws_for_limit_zero(self) -> None:
        """Fail call with limit=0."""
        # Prepare
        state: State = self.state_factory()

        # Execute & Assert
        with self.assertRaises(AssertionError):
            state.get_task_ins(node_id=1, limit=0)

    # TaskRes tests
    def test_task_res_store_and_retrieve_by_task_ins_id(self) -> None:
        """Store TaskRes retrieve it by task_ins_id."""
        # Prepare
        state: State = self.state_factory()
        task_ins_id = uuid4()
        task_res = create_task_res(
            producer_node_id=0, anonymous=True, ancestry=[str(task_ins_id)]
        )

        # Execute
        task_res_uuid = state.store_task_res(task_res)
        task_res_list = state.get_task_res(task_ids={task_ins_id}, limit=None)

        # Assert
        retrieved_task_res = task_res_list[0]
        assert retrieved_task_res.task_id == str(task_res_uuid)

    def test_node_ids_initial_state(self) -> None:
        """Test retrieving all node_ids and empty initial state."""
        # Prepare
        state: State = self.state_factory()

        # Execute
        retrieved_node_ids = state.get_nodes()

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_register_node_and_get_nodes(self) -> None:
        """Test registering a client node."""
        # Prepare
        state: State = self.state_factory()
        node_ids = list(range(1, 11))

        # Execute
        for i in node_ids:
            state.register_node(i)
        retrieved_node_ids = state.get_nodes()

        # Assert
        for i in retrieved_node_ids:
            assert i in node_ids

    def test_unregister_node(self) -> None:
        """Test unregistering a client node."""
        # Prepare
        state: State = self.state_factory()
        node_id = 2

        # Execute
        state.register_node(node_id)
        state.unregister_node(node_id)
        retrieved_node_ids = state.get_nodes()

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_num_task_ins(self) -> None:
        """Test if num_tasks returns correct number of not delivered task_ins."""
        # Prepare
        state: State = self.state_factory()
        task_0 = create_task_ins(consumer_node_id=0, anonymous=True)
        task_1 = create_task_ins(consumer_node_id=0, anonymous=True)

        # Store two tasks
        state.store_task_ins(task_0)
        state.store_task_ins(task_1)

        # Execute
        num = state.num_task_ins()

        # Assert
        assert num == 2

    def test_num_task_res(self) -> None:
        """Test if num_tasks returns correct number of not delivered task_res."""
        # Prepare
        state: State = self.state_factory()
        task_0 = create_task_res(producer_node_id=0, anonymous=True, ancestry=["1"])
        task_1 = create_task_res(producer_node_id=0, anonymous=True, ancestry=["1"])

        # Store two tasks
        state.store_task_res(task_0)
        state.store_task_res(task_1)

        # Execute
        num = state.num_task_res()

        # Assert
        assert num == 2


def create_task_ins(
    consumer_node_id: int, anonymous: bool, delivered_at: str = ""
) -> TaskIns:
    """Create a TaskIns for testing."""
    consumer = Node(
        node_id=consumer_node_id,
        anonymous=anonymous,
    )
    task = TaskIns(
        task_id="",
        group_id="",
        workload_id="",
        task=Task(
            delivered_at=delivered_at,
            producer=Node(node_id=0, anonymous=True),
            consumer=consumer,
            legacy_server_message=ServerMessage(
                reconnect_ins=ServerMessage.ReconnectIns()
            ),
        ),
    )
    return task


def create_task_res(
    producer_node_id: int, anonymous: bool, ancestry: List[str]
) -> TaskRes:
    """Create a TaskRes for testing."""
    task_res = TaskRes(
        task_id="",
        group_id="",
        workload_id="",
        task=Task(
            producer=Node(node_id=producer_node_id, anonymous=anonymous),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=ancestry,
            legacy_client_message=ClientMessage(
                disconnect_res=ClientMessage.DisconnectRes()
            ),
        ),
    )
    return task_res


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> State:
        """Return InMemoryState."""
        return InMemoryState()


class SqliteInMemoryStateTest(StateTest, unittest.TestCase):
    """Test SqliteState implemenation with in-memory database."""

    __test__ = True

    def state_factory(self) -> SqliteState:
        """Return SqliteState with in-memory database."""
        state = SqliteState(":memory:")
        state.initialize()
        return state

    def test_initialize(self) -> None:
        """Test initialization."""
        # Prepare
        state = self.state_factory()

        # Execute
        result = state.query("SELECT name FROM sqlite_schema;")

        # Assert
        assert len(result) == 6


class SqliteFileBasedTest(StateTest, unittest.TestCase):
    """Test SqliteState implemenation with file-based database."""

    __test__ = True

    def state_factory(self) -> SqliteState:
        """Return SqliteState with file-based database."""
        # pylint: disable-next=consider-using-with,attribute-defined-outside-init
        self.tmp_file = tempfile.NamedTemporaryFile()
        state = SqliteState(database_path=self.tmp_file.name)
        state.initialize()
        return state

    def test_initialize(self) -> None:
        """Test initialization."""
        # Prepare
        state = self.state_factory()

        # Execute
        result = state.query("SELECT name FROM sqlite_schema;")

        # Assert
        assert len(result) == 6


if __name__ == "__main__":
    unittest.main(verbosity=2)

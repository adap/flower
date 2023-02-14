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
import tempfile
import random
from google.protobuf.json_format import MessageToJson

from typing import List

from .in_memory_state import InMemoryState
from .sqlite_state import SqliteState
from .state import State

from datetime import datetime, timezone
from uuid import uuid4

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes


# class StateTest(unittest.TestCase):
class StateTest:
    """Test all state implementations."""

    @property
    def state_factory(self) -> State:
        raise NotImplementedError()

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

    ############################### Init tests ################################
    def test_init_state(self) -> None:
        """Test that state is innitialized correctly."""

        # Execute
        state = self.state_factory()

        # Assert
        assert isinstance(state, State)

    ########################### TaskIns tests ################################
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
        retrieved_task_ins = task_ins_list[0]
        assert retrieved_task_ins.task_id == str(task_ins_uuid)

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
        task_ins = create_task_ins(
            consumer_node_id=1, anonymous=False, delivered_at="1989-11-09"
        )

        # Execute
        _ = state.store_task_ins(task_ins)
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

    ########################### TaskRes tests ################################
    def test_store_task_res_fail(self) -> None:
        """Fail storeing task_ins because of missing ancestry."""
        # Prepare
        state: State = self.state_factory()
        task_res = create_task_res(ancestry=[])

        # Execute
        empty_result = state.store_task_res(task_res)

        # Assert
        assert empty_result is None

    def test_task_res_retrieve_by_task_ins_id(self) -> None:
        """Store TaskRes retrieve it by task_ins_id."""
        # Prepare
        state: State = self.state_factory()
        task_ins_id = uuid4()
        task_res = create_task_res(ancestry=[str(task_ins_id)])

        # Execute
        task_res_uuid = state.store_task_res(task_res)
        task_res_list = state.get_task_res(task_ids=set([task_ins_id]), limit=None)

        # Assert
        retrieved_task_res = task_res_list[0]
        assert retrieved_task_res.task_id == str(task_res_uuid)

    # def test_task_res_store_identity_and_fail_retrieving_anonymous(self) -> None:
    #     """Store identity TaskIns and fail retrieving it as anonymous."""
    #     # Prepare
    #     state: State = self.state_factory()
    #     task_res = create_task_res(ancestry=[])

    #     # Execute
    #     _ = state.store_task_res(task_res)
    #     task_res_list = state.get_task_res(task_ids=set(), limit=None)

    #     # Assert
    #     assert len(task_res_list) == 0

    # def test_task_res_store_identity_and_retrieve_identity(self) -> None:
    #     """Store identity TaskIns and retrieve it."""
    #     # Prepare
    #     state: State = self.state_factory()
    #     task_res = create_task_res(ancestry=[])

    #     # Execute
    #     task_res_uuid = state.store_task_res(task_res)
    #     task_res_list = state.get_task_res(task_ids=set(), limit=None)

    #     # Assert
    #     assert len(task_res_list) == 1

    #     retrieved_task_res = task_res_list[0]
    #     assert retrieved_task_res.task_id == str(task_res_uuid)

    # def test_task_res_store_delivered_and_fail_retrieving(self) -> None:
    #     """Fail retrieving delivered task."""
    #     # Prepare
    #     state: State = self.state_factory()
    #     task_res = create_task_res(
    #         consumer_node_id=1, anonymous=False, delivered_at="1989-11-09"
    #     )

    #     # Execute
    #     _ = state.store_task_res(task_res)
    #     task_res_list = state.get_task_res(task_ids=set(), limit=None)

    #     # Assert
    #     assert len(task_res_list) == 0

    # def test_get_task_res_limit_throws_for_limit_zero(self) -> None:
    #     """Fail call with limit=0."""
    #     # Prepare
    #     state: State = self.state_factory()

    #     # Execute
    #     with self.assertRaises(AssertionError):
    #         state.get_task_res(task_ids=set(), limit=0)

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
            assert 10 in node_ids

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
        ),
    )
    return task


def create_task_res(ancestry: List[str]) -> TaskIns:
    """Create a TaskRes for testing."""
    task_res = TaskRes(
        task_id=str(uuid4()),
        group_id="",
        workload_id="",
        task=Task(
            consumer=Node(node_id=0, anonymous=True),
            ancestry=ancestry,
        ),
    )
    return task_res


def printpb(msg):
    print(
        MessageToJson(
            msg,
            including_default_value_fields=True,
        )
    )


class InMemoryStateTest(StateTest, unittest.TestCase):
    state_factory = lambda _: InMemoryState()


# class SqliteInMemoryStateTest(StateTest, unittest.TestCase):
#     state_factory = lambda _: SqliteState()


# def sql_lite_file_based_state_factory(_) -> State:
#     file_path = tempfile.TemporaryFile()
#     return SqliteState(database_path=file_path)


# class SqliteFileBaseTest(StateTest, unittest.TestCase):
#     state_factory = sql_lite_file_based_state_factory


if __name__ == "__main__":
    unittest.main(verbosity=2)

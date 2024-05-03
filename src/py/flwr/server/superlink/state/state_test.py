# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
# pylint: disable=invalid-name, disable=R0904

import tempfile
import time
import unittest
from abc import abstractmethod
from datetime import datetime, timezone
from typing import List
from unittest.mock import patch
from uuid import uuid4

from flwr.common import DEFAULT_TTL
from flwr.common.constant import ErrorCode
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_key_pairs,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.recordset_pb2 import RecordSet  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.superlink.state import InMemoryState, SqliteState, State


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def state_factory(self) -> State:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_create_and_get_run(self) -> None:
        """Test if create_run and get_run work correctly."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("Mock/mock", "v1.0.0")

        # Execute
        actual_run_id, fab_id, fab_version = state.get_run(run_id)

        # Assert
        assert actual_run_id == run_id
        assert fab_id == "Mock/mock"
        assert fab_version == "v1.0.0"

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
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(
            consumer_node_id=consumer_node_id, anonymous=False, run_id=run_id
        )

        assert task_ins.task.created_at < time.time()  # pylint: disable=no-member
        assert task_ins.task.delivered_at == ""  # pylint: disable=no-member

        # Execute
        state.store_task_ins(task_ins=task_ins)
        task_ins_list = state.get_task_ins(node_id=consumer_node_id, limit=10)

        # Assert
        assert len(task_ins_list) == 1

        actual_task_ins = task_ins_list[0]

        assert actual_task_ins.task_id == task_ins.task_id  # pylint: disable=no-member
        assert actual_task_ins.HasField("task")

        actual_task = actual_task_ins.task

        assert actual_task.delivered_at != ""

        assert actual_task.created_at < actual_task.pushed_at
        assert datetime.fromisoformat(actual_task.delivered_at) > datetime(
            2020, 1, 1, tzinfo=timezone.utc
        )
        assert actual_task.ttl > 0

    def test_store_and_delete_tasks(self) -> None:
        """Test delete_tasks."""
        # Prepare
        consumer_node_id = 1
        state = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins_0 = create_task_ins(
            consumer_node_id=consumer_node_id, anonymous=False, run_id=run_id
        )
        task_ins_1 = create_task_ins(
            consumer_node_id=consumer_node_id, anonymous=False, run_id=run_id
        )
        task_ins_2 = create_task_ins(
            consumer_node_id=consumer_node_id, anonymous=False, run_id=run_id
        )

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
            producer_node_id=100,
            anonymous=False,
            ancestry=[str(task_id_0)],
            run_id=run_id,
        )

        _ = state.store_task_res(task_res=task_res_0)
        _ = state.get_task_res(task_ids={task_id_0}, limit=None)

        # Insert one TaskRes, but don't retrive it
        task_res_1: TaskRes = create_task_res(
            producer_node_id=100,
            anonymous=False,
            ancestry=[str(task_id_1)],
            run_id=run_id,
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
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)

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
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_fail_retrieving_anonymous(self) -> None:
        """Store identity TaskIns and fail retrieving it as anonymous."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False, run_id=run_id)

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=None, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_retrieve_identity(self) -> None:
        """Store identity TaskIns and retrieve it."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False, run_id=run_id)

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
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins = create_task_ins(consumer_node_id=1, anonymous=False, run_id=run_id)

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

    def test_task_ins_store_invalid_run_id_and_fail(self) -> None:
        """Store TaskIns with invalid run_id and fail."""
        # Prepare
        state: State = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=61016)

        # Execute
        task_id = state.store_task_ins(task_ins)

        # Assert
        assert task_id is None

    # TaskRes tests
    def test_task_res_store_and_retrieve_by_task_ins_id(self) -> None:
        """Store TaskRes retrieve it by task_ins_id."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_ins_id = uuid4()
        task_res = create_task_res(
            producer_node_id=0,
            anonymous=True,
            ancestry=[str(task_ins_id)],
            run_id=run_id,
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
        run_id = state.create_run("mock/mock", "v1.0.0")

        # Execute
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_create_node_and_get_nodes(self) -> None:
        """Test creating a client node."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_ids = []

        # Execute
        for _ in range(10):
            node_ids.append(state.create_node(ping_interval=10))
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        for i in retrieved_node_ids:
            assert i in node_ids

    def test_create_node_public_key(self) -> None:
        """Test creating a client node with public key."""
        # Prepare
        state: State = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run("mock/mock", "v1.0.0")

        # Execute
        node_id = state.create_node(ping_interval=10, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        # Assert
        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id == node_id
    
    def test_create_node_public_key_twice(self) -> None:
        """Test creating a client node with public key."""
        # Prepare
        state: State = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_id = state.create_node(ping_interval=10, public_key=public_key)

        # Execute
        new_node_id = state.create_node(ping_interval=10, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)
        
        # Assert
        assert new_node_id == 0
        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id == node_id
        
        if isinstance(state, InMemoryState):
            assert len(state.node_ids) == 1
            assert len(state.public_key_to_node_id) == 1

    def test_delete_node(self) -> None:
        """Test deleting a client node."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_id = state.create_node(ping_interval=10)

        # Execute
        state.delete_node(node_id)
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_delete_node_public_key(self) -> None:
        """Test deleting a client node with public key."""
        # Prepare
        state: State = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_id = state.create_node(ping_interval=10, public_key=public_key)

        # Execute
        state.delete_node(node_id, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        # Assert
        assert len(retrieved_node_ids) == 0
        assert retrieved_node_id is None

    def test_get_nodes_invalid_run_id(self) -> None:
        """Test retrieving all node_ids with invalid run_id."""
        # Prepare
        state: State = self.state_factory()
        state.create_run("mock/mock", "v1.0.0")
        invalid_run_id = 61016
        state.create_node(ping_interval=10)

        # Execute
        retrieved_node_ids = state.get_nodes(invalid_run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_num_task_ins(self) -> None:
        """Test if num_tasks returns correct number of not delivered task_ins."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_0 = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)
        task_1 = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)

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
        run_id = state.create_run("mock/mock", "v1.0.0")
        task_0 = create_task_res(
            producer_node_id=0, anonymous=True, ancestry=["1"], run_id=run_id
        )
        task_1 = create_task_res(
            producer_node_id=0, anonymous=True, ancestry=["1"], run_id=run_id
        )

        # Store two tasks
        state.store_task_res(task_0)
        state.store_task_res(task_1)

        # Execute
        num = state.num_task_res()

        # Assert
        assert num == 2

    def test_server_private_public_key(self) -> None:
        """Test get server private and public key after inserting."""
        # Prepare
        state: State = self.state_factory()
        private_key, public_key = generate_key_pairs()
        private_key_bytes = private_key_to_bytes(private_key)
        public_key_bytes = public_key_to_bytes(public_key)

        # Execute
        state.store_server_private_public_key(private_key_bytes, public_key_bytes)
        server_private_key = state.get_server_private_key()
        server_public_key = state.get_server_public_key()

        # Assert
        assert server_private_key == private_key_bytes
        assert server_public_key == public_key_bytes

    def test_server_private_public_key_none(self) -> None:
        """Test get server private and public key without inserting."""
        # Prepare
        state: State = self.state_factory()

        # Execute
        server_private_key = state.get_server_private_key()
        server_public_key = state.get_server_public_key()

        # Assert
        assert server_private_key is None
        assert server_public_key is None

    def test_store_server_private_public_key_twice(self) -> None:
        """Test inserting private and public key twice."""
        # Prepare
        state: State = self.state_factory()
        private_key, public_key = generate_key_pairs()
        private_key_bytes = private_key_to_bytes(private_key)
        public_key_bytes = public_key_to_bytes(public_key)
        new_private_key, new_public_key = generate_key_pairs()
        new_private_key_bytes = private_key_to_bytes(new_private_key)
        new_public_key_bytes = public_key_to_bytes(new_public_key)

        # Execute
        state.store_server_private_public_key(private_key_bytes, public_key_bytes)

        # Assert
        with self.assertRaises(RuntimeError):
            state.store_server_private_public_key(
                new_private_key_bytes, new_public_key_bytes
            )

    def test_client_public_keys(self) -> None:
        """Test store_client_public_keys and get_client_public_keys from state."""
        # Prepare
        state: State = self.state_factory()
        key_pairs = [generate_key_pairs() for _ in range(3)]
        public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

        # Execute
        state.store_client_public_keys(public_keys)
        client_public_keys = state.get_client_public_keys()

        # Assert
        assert client_public_keys == public_keys

    def test_client_public_key(self) -> None:
        """Test store_client_public_key and get_client_public_keys from state."""
        # Prepare
        state: State = self.state_factory()
        key_pairs = [generate_key_pairs() for _ in range(3)]
        public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

        # Execute
        for public_key in public_keys:
            state.store_client_public_key(public_key)
        client_public_keys = state.get_client_public_keys()

        # Assert
        assert client_public_keys == public_keys

    def test_acknowledge_ping(self) -> None:
        """Test if acknowledge_ping works and if get_nodes return online nodes."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_ids = [state.create_node(ping_interval=10) for _ in range(100)]
        for node_id in node_ids[:70]:
            state.acknowledge_ping(node_id, ping_interval=30)
        for node_id in node_ids[70:]:
            state.acknowledge_ping(node_id, ping_interval=90)

        # Execute
        current_time = time.time()
        with patch("time.time", side_effect=lambda: current_time + 50):
            actual_node_ids = state.get_nodes(run_id)

        # Assert
        self.assertSetEqual(actual_node_ids, set(node_ids[70:]))

    def test_node_unavailable_error(self) -> None:
        """Test if get_task_res return TaskRes containing node unavailable error."""
        # Prepare
        state: State = self.state_factory()
        run_id = state.create_run("mock/mock", "v1.0.0")
        node_id_0 = state.create_node(ping_interval=90)
        node_id_1 = state.create_node(ping_interval=30)
        # Create and store TaskIns
        task_ins_0 = create_task_ins(
            consumer_node_id=node_id_0, anonymous=False, run_id=run_id
        )
        task_ins_1 = create_task_ins(
            consumer_node_id=node_id_1, anonymous=False, run_id=run_id
        )
        task_id_0 = state.store_task_ins(task_ins=task_ins_0)
        task_id_1 = state.store_task_ins(task_ins=task_ins_1)
        assert task_id_0 is not None and task_id_1 is not None

        # Get TaskIns to mark them delivered
        state.get_task_ins(node_id=node_id_0, limit=None)

        # Create and store TaskRes
        task_res_0 = create_task_res(
            producer_node_id=100,
            anonymous=False,
            ancestry=[str(task_id_0)],
            run_id=run_id,
        )
        state.store_task_res(task_res_0)

        # Execute
        current_time = time.time()
        task_res_list: List[TaskRes] = []
        with patch("time.time", side_effect=lambda: current_time + 50):
            task_res_list = state.get_task_res({task_id_0, task_id_1}, limit=None)

        # Assert
        assert len(task_res_list) == 2
        err_taskres = task_res_list[1]
        assert err_taskres.task.HasField("error")
        assert err_taskres.task.error.code == ErrorCode.NODE_UNAVAILABLE


def create_task_ins(
    consumer_node_id: int,
    anonymous: bool,
    run_id: int,
    delivered_at: str = "",
) -> TaskIns:
    """Create a TaskIns for testing."""
    consumer = Node(
        node_id=consumer_node_id,
        anonymous=anonymous,
    )
    task = TaskIns(
        task_id="",
        group_id="",
        run_id=run_id,
        task=Task(
            delivered_at=delivered_at,
            producer=Node(node_id=0, anonymous=True),
            consumer=consumer,
            task_type="mock",
            recordset=RecordSet(parameters={}, metrics={}, configs={}),
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )
    task.task.pushed_at = time.time()
    return task


def create_task_res(
    producer_node_id: int,
    anonymous: bool,
    ancestry: List[str],
    run_id: int,
) -> TaskRes:
    """Create a TaskRes for testing."""
    task_res = TaskRes(
        task_id="",
        group_id="",
        run_id=run_id,
        task=Task(
            producer=Node(node_id=producer_node_id, anonymous=anonymous),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=ancestry,
            task_type="mock",
            recordset=RecordSet(parameters={}, metrics={}, configs={}),
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )
    task_res.task.pushed_at = time.time()
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
        assert len(result) == 13


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
        assert len(result) == 13


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
"""Tests all LinkState implemenations have to conform to."""
# pylint: disable=invalid-name, too-many-lines, R0904, R0913

import tempfile
import time
import unittest
from abc import abstractmethod
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID

from parameterized import parameterized

from flwr.common import DEFAULT_TTL, ConfigsRecord, Context, RecordSet, now
from flwr.common.constant import Status, SubStatus
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_key_pairs,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.common.typing import RunStatus

# pylint: disable=E0611
from flwr.proto.node_pb2 import Node
from flwr.proto.recordset_pb2 import RecordSet as ProtoRecordSet
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes

# pylint: enable=E0611
from flwr.server.superlink.linkstate import (
    InMemoryLinkState,
    LinkState,
    SqliteLinkState,
)


class StateTest(unittest.TestCase):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def state_factory(self) -> LinkState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_create_and_get_run(self) -> None:
        """Test if create_run and get_run work correctly."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )

        # Execute
        run = state.get_run(run_id)

        # Assert
        assert run is not None
        assert run.run_id == run_id
        assert run.fab_hash == "9f86d08"
        assert run.override_config["test_key"] == "test_value"

    def test_get_all_run_ids(self) -> None:
        """Test if get_run_ids works correctly."""
        # Prepare
        state = self.state_factory()
        run_id1 = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )
        run_id2 = state.create_run(
            None, None, "fffffff", {"mock_key": "mock_value"}, ConfigsRecord()
        )

        # Execute
        run_ids = state.get_run_ids()

        # Assert
        assert run_id1 in run_ids
        assert run_id2 in run_ids

    def test_get_all_run_ids_empty(self) -> None:
        """Test if get_run_ids works correctly when no runs are present."""
        # Prepare
        state = self.state_factory()

        # Execute
        run_ids = state.get_run_ids()

        # Assert
        assert len(run_ids) == 0

    def test_get_pending_run_id(self) -> None:
        """Test if get_pending_run_id works correctly."""
        # Prepare
        state = self.state_factory()
        _ = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )
        run_id2 = state.create_run(
            None, None, "fffffff", {"mock_key": "mock_value"}, ConfigsRecord()
        )
        state.update_run_status(run_id2, RunStatus(Status.STARTING, "", ""))

        # Execute
        pending_run_id = state.get_pending_run_id()
        assert pending_run_id is not None
        run_status_dict = state.get_run_status({pending_run_id})
        assert run_status_dict[pending_run_id].status == Status.PENDING

        # Change state
        state.update_run_status(pending_run_id, RunStatus(Status.STARTING, "", ""))
        # Attempt get pending run
        pending_run_id = state.get_pending_run_id()
        assert pending_run_id is None

    def test_get_and_update_run_status(self) -> None:
        """Test if get_run_status and update_run_status work correctly."""
        # Prepare
        state = self.state_factory()
        run_id1 = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )
        run_id2 = state.create_run(
            None, None, "fffffff", {"mock_key": "mock_value"}, ConfigsRecord()
        )
        state.update_run_status(run_id2, RunStatus(Status.STARTING, "", ""))
        state.update_run_status(run_id2, RunStatus(Status.RUNNING, "", ""))

        # Execute
        run_status_dict = state.get_run_status({run_id1, run_id2})
        status1 = run_status_dict[run_id1]
        status2 = run_status_dict[run_id2]

        # Assert
        assert status1.status == Status.PENDING
        assert status2.status == Status.RUNNING

    @parameterized.expand([(0,), (1,), (2,)])  # type: ignore
    def test_status_transition_valid(
        self, num_transitions_before_finishing: int
    ) -> None:
        """Test valid run status transactions."""
        # Prepare
        state = self.state_factory()
        run_id = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )

        # Execute and assert
        status = state.get_run_status({run_id})[run_id]
        assert status.status == Status.PENDING

        if num_transitions_before_finishing > 0:
            assert state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
            status = state.get_run_status({run_id})[run_id]
            assert status.status == Status.STARTING

        if num_transitions_before_finishing > 1:
            assert state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
            status = state.get_run_status({run_id})[run_id]
            assert status.status == Status.RUNNING

        assert state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.FAILED, "mock failure")
        )

        status = state.get_run_status({run_id})[run_id]
        assert status.status == Status.FINISHED

    def test_status_transition_invalid(self) -> None:
        """Test invalid run status transitions."""
        # Prepare
        state = self.state_factory()
        run_id = state.create_run(
            None, None, "9f86d08", {"test_key": "test_value"}, ConfigsRecord()
        )
        run_statuses = [
            RunStatus(Status.PENDING, "", ""),
            RunStatus(Status.STARTING, "", ""),
            RunStatus(Status.PENDING, "", ""),
            RunStatus(Status.FINISHED, SubStatus.COMPLETED, ""),
        ]

        # Execute and assert
        # Cannot transition from RunStatus.PENDING to RunStatus.PENDING,
        # RunStatus.RUNNING, or RunStatus.FINISHED with COMPLETED substatus
        for run_status in [s for s in run_statuses if s.status != Status.STARTING]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        # Cannot transition from RunStatus.STARTING to RunStatus.PENDING,
        # RunStatus.STARTING, or RunStatus.FINISHED with COMPLETED substatus
        for run_status in [s for s in run_statuses if s.status != Status.RUNNING]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        # Cannot transition from RunStatus.RUNNING
        # to RunStatus.PENDING, RunStatus.STARTING, or RunStatus.RUNNING
        for run_status in [s for s in run_statuses if s.status != Status.FINISHED]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
        )
        # Cannot transition to any status from RunStatus.FINISHED
        run_statuses += [
            RunStatus(Status.FINISHED, SubStatus.FAILED, ""),
            RunStatus(Status.FINISHED, SubStatus.STOPPED, ""),
        ]
        for run_status in run_statuses:
            assert not state.update_run_status(run_id, run_status)

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
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        assert task_ins.task.created_at < time.time()  # pylint: disable=no-member
        assert task_ins.task.delivered_at == ""  # pylint: disable=no-member

        # Execute
        state.store_task_ins(task_ins=task_ins)
        task_ins_list = state.get_task_ins(node_id=node_id, limit=10)

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

    def test_store_task_ins_invalid_node_id(self) -> None:
        """Test store_task_ins with invalid node_id."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        invalid_node_id = 61016 if node_id != 61016 else 61017
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=invalid_node_id, anonymous=False, run_id=run_id
        )
        task_ins2 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins2.task.producer.node_id = 61016

        # Execute and assert
        assert state.store_task_ins(task_ins) is None
        assert state.store_task_ins(task_ins2) is None

    def test_store_and_delete_tasks(self) -> None:
        """Test delete_tasks."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins_0 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins_1 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins_2 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        # Insert three TaskIns
        task_id_0 = state.store_task_ins(task_ins=task_ins_0)
        task_id_1 = state.store_task_ins(task_ins=task_ins_1)
        task_id_2 = state.store_task_ins(task_ins=task_ins_2)

        assert task_id_0
        assert task_id_1
        assert task_id_2

        # Get TaskIns to mark them delivered
        _ = state.get_task_ins(node_id=node_id, limit=None)

        # Insert one TaskRes and retrive it to mark it as delivered
        task_res_0 = create_task_res(
            producer_node_id=node_id,
            anonymous=False,
            ancestry=[str(task_id_0)],
            run_id=run_id,
        )

        _ = state.store_task_res(task_res=task_res_0)
        _ = state.get_task_res(task_ids={task_id_0})

        # Insert one TaskRes, but don't retrive it
        task_res_1: TaskRes = create_task_res(
            producer_node_id=node_id,
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

        state.delete_tasks({task_id_0})
        assert state.num_task_ins() == 2
        assert state.num_task_res() == 1

        state.delete_tasks({task_id_1})
        assert state.num_task_ins() == 1
        assert state.num_task_res() == 0

        state.delete_tasks({task_id_2})
        assert state.num_task_ins() == 0
        assert state.num_task_res() == 0

    def test_get_task_ids_from_run_id(self) -> None:
        """Test get_task_ids_from_run_id."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id_0 = state.create_run(None, None, "8g13kl7", {}, ConfigsRecord())
        # Insert tasks with the same run_id
        task_ins_0 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id_0
        )
        task_ins_1 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id_0
        )
        # Insert a task with a different run_id to ensure it does not appear in result
        run_id_1 = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins_2 = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id_1
        )

        # Insert three TaskIns
        task_id_0 = state.store_task_ins(task_ins=task_ins_0)
        task_id_1 = state.store_task_ins(task_ins=task_ins_1)
        task_id_2 = state.store_task_ins(task_ins=task_ins_2)

        assert task_id_0
        assert task_id_1
        assert task_id_2

        expected_task_ids = {task_id_0, task_id_1}

        # Execute
        result = state.get_task_ids_from_run_id(run_id_0)
        bad_result = state.get_task_ids_from_run_id(15)

        self.assertEqual(len(bad_result), 0)
        self.assertSetEqual(result, expected_task_ids)

    # Init tests
    def test_init_state(self) -> None:
        """Test that state is initialized correctly."""
        # Execute
        state = self.state_factory()

        # Assert
        assert isinstance(state, LinkState)

    # TaskIns tests
    def test_task_ins_store_anonymous_and_retrieve_anonymous(self) -> None:
        """Store one TaskIns.

        Create anonymous task and retrieve it.
        """
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
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
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_fail_retrieving_anonymous(self) -> None:
        """Store identity TaskIns and fail retrieving it as anonymous."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        # Execute
        _ = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=None, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_task_ins_store_identity_and_retrieve_identity(self) -> None:
        """Store identity TaskIns and retrieve it."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        # Execute
        task_ins_uuid = state.store_task_ins(task_ins)
        task_ins_list = state.get_task_ins(node_id=node_id, limit=None)

        # Assert
        assert len(task_ins_list) == 1

        retrieved_task_ins = task_ins_list[0]
        assert retrieved_task_ins.task_id == str(task_ins_uuid)

    def test_task_ins_store_delivered_and_fail_retrieving(self) -> None:
        """Fail retrieving delivered task."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        # Execute
        _ = state.store_task_ins(task_ins)

        # 1st get: set to delivered
        task_ins_list = state.get_task_ins(node_id=node_id, limit=None)

        assert len(task_ins_list) == 1

        # 2nd get: no TaskIns because it was already delivered before
        task_ins_list = state.get_task_ins(node_id=1, limit=None)

        # Assert
        assert len(task_ins_list) == 0

    def test_get_task_ins_limit_throws_for_limit_zero(self) -> None:
        """Fail call with limit=0."""
        # Prepare
        state: LinkState = self.state_factory()

        # Execute & Assert
        with self.assertRaises(AssertionError):
            state.get_task_ins(node_id=1, limit=0)

    def test_task_ins_store_invalid_run_id_and_fail(self) -> None:
        """Store TaskIns with invalid run_id and fail."""
        # Prepare
        state: LinkState = self.state_factory()
        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=61016)

        # Execute
        task_id = state.store_task_ins(task_ins)

        # Assert
        assert task_id is None

    # TaskRes tests
    def test_task_res_store_and_retrieve_by_task_ins_id(self) -> None:
        """Store TaskRes retrieve it by task_ins_id."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)
        task_ins_id = state.store_task_ins(task_ins)

        task_res = create_task_res(
            producer_node_id=0,
            anonymous=True,
            ancestry=[str(task_ins_id)],
            run_id=run_id,
        )

        # Execute
        task_res_uuid = state.store_task_res(task_res)

        assert task_ins_id
        task_res_list = state.get_task_res(task_ids={task_ins_id})

        # Assert
        retrieved_task_res = task_res_list[0]
        assert retrieved_task_res.task_id == str(task_res_uuid)

    def test_node_ids_initial_state(self) -> None:
        """Test retrieving all node_ids and empty initial state."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        # Execute
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_create_node_and_get_nodes(self) -> None:
        """Test creating a client node."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
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
        state: LinkState = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        # Execute
        node_id = state.create_node(ping_interval=10, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        # Assert
        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id == node_id

    def test_create_node_public_key_twice(self) -> None:
        """Test creating a client node with same public key twice."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        node_id = state.create_node(ping_interval=10, public_key=public_key)

        # Execute
        new_node_id = state.create_node(ping_interval=10, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        # Assert
        assert new_node_id == 0
        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id == node_id

        # Assert node_ids and public_key_to_node_id are synced
        if isinstance(state, InMemoryLinkState):
            assert len(state.node_ids) == 1
            assert len(state.public_key_to_node_id) == 1

    def test_delete_node(self) -> None:
        """Test deleting a client node."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        node_id = state.create_node(ping_interval=10)

        # Execute
        state.delete_node(node_id)
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_delete_node_public_key(self) -> None:
        """Test deleting a client node with public key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        node_id = state.create_node(ping_interval=10, public_key=public_key)

        # Execute
        state.delete_node(node_id, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        # Assert
        assert len(retrieved_node_ids) == 0
        assert retrieved_node_id is None

    def test_delete_node_public_key_none(self) -> None:
        """Test deleting a client node with public key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        node_id = 0

        # Execute & Assert
        with self.assertRaises(ValueError):
            state.delete_node(node_id, public_key=public_key)

        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        assert len(retrieved_node_ids) == 0
        assert retrieved_node_id is None

    def test_delete_node_wrong_public_key(self) -> None:
        """Test deleting a client node with wrong public key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        wrong_public_key = b"mock_mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        node_id = state.create_node(ping_interval=10, public_key=public_key)

        # Execute & Assert
        with self.assertRaises(ValueError):
            state.delete_node(node_id, public_key=wrong_public_key)

        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(public_key)

        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id == node_id

    def test_get_node_id_wrong_public_key(self) -> None:
        """Test retrieving a client node with wrong public key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        wrong_public_key = b"mock_mock"
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        # Execute
        state.create_node(ping_interval=10, public_key=public_key)
        retrieved_node_ids = state.get_nodes(run_id)
        retrieved_node_id = state.get_node_id(wrong_public_key)

        # Assert
        assert len(retrieved_node_ids) == 1
        assert retrieved_node_id is None

    def test_get_nodes_invalid_run_id(self) -> None:
        """Test retrieving all node_ids with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        invalid_run_id = 61016
        state.create_node(ping_interval=10)

        # Execute
        retrieved_node_ids = state.get_nodes(invalid_run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_num_task_ins(self) -> None:
        """Test if num_tasks returns correct number of not delivered task_ins."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
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
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        task_ins_0 = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)
        task_ins_1 = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)
        task_ins_id_0 = state.store_task_ins(task_ins_0)
        task_ins_id_1 = state.store_task_ins(task_ins_1)

        task_0 = create_task_res(
            producer_node_id=0,
            anonymous=True,
            ancestry=[str(task_ins_id_0)],
            run_id=run_id,
        )
        task_1 = create_task_res(
            producer_node_id=0,
            anonymous=True,
            ancestry=[str(task_ins_id_1)],
            run_id=run_id,
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
        state: LinkState = self.state_factory()
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
        state: LinkState = self.state_factory()

        # Execute
        server_private_key = state.get_server_private_key()
        server_public_key = state.get_server_public_key()

        # Assert
        assert server_private_key is None
        assert server_public_key is None

    def test_store_server_private_public_key_twice(self) -> None:
        """Test inserting private and public key twice."""
        # Prepare
        state: LinkState = self.state_factory()
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

    def test_node_public_keys(self) -> None:
        """Test store_node_public_keys and get_node_public_keys from state."""
        # Prepare
        state: LinkState = self.state_factory()
        key_pairs = [generate_key_pairs() for _ in range(3)]
        public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

        # Execute
        state.store_node_public_keys(public_keys)
        node_public_keys = state.get_node_public_keys()

        # Assert
        assert node_public_keys == public_keys

    def test_node_public_key(self) -> None:
        """Test store_node_public_key and get_node_public_keys from state."""
        # Prepare
        state: LinkState = self.state_factory()
        key_pairs = [generate_key_pairs() for _ in range(3)]
        public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

        # Execute
        for public_key in public_keys:
            state.store_node_public_key(public_key)
        node_public_keys = state.get_node_public_keys()

        # Assert
        assert node_public_keys == public_keys

    def test_acknowledge_ping(self) -> None:
        """Test if acknowledge_ping works and if get_nodes return online nodes."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
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

    def test_store_task_res_task_ins_expired(self) -> None:
        """Test behavior of store_task_res when the TaskIns it references is expired."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        task_ins = create_task_ins(consumer_node_id=0, anonymous=True, run_id=run_id)
        task_ins.task.created_at = time.time() - task_ins.task.ttl + 0.5
        task_ins_id = state.store_task_ins(task_ins)

        with patch(
            "time.time",
            side_effect=lambda: task_ins.task.created_at + task_ins.task.ttl + 0.1,
        ):  # Expired by 0.1 seconds
            task = create_task_res(
                producer_node_id=0,
                anonymous=True,
                ancestry=[str(task_ins_id)],
                run_id=run_id,
            )

            # Execute
            result = state.store_task_res(task)

        # Assert
        assert result is None

    def test_store_task_res_limit_ttl(self) -> None:
        """Test the behavior of store_task_res regarding the TTL limit of TaskRes."""
        current_time = time.time()

        test_cases = [
            (
                current_time - 5,
                10,
                current_time - 2,
                6,
                True,
            ),  # TaskRes within allowed TTL
            (
                current_time - 5,
                10,
                current_time - 2,
                15,
                False,
            ),  # TaskRes TTL exceeds max allowed TTL
        ]

        for (
            task_ins_created_at,
            task_ins_ttl,
            task_res_created_at,
            task_res_ttl,
            expected_store_result,
        ) in test_cases:

            # Prepare
            state: LinkState = self.state_factory()
            run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

            task_ins = create_task_ins(
                consumer_node_id=0, anonymous=True, run_id=run_id
            )
            task_ins.task.created_at = task_ins_created_at
            task_ins.task.ttl = task_ins_ttl
            task_ins_id = state.store_task_ins(task_ins)

            task_res = create_task_res(
                producer_node_id=0,
                anonymous=True,
                ancestry=[str(task_ins_id)],
                run_id=run_id,
            )
            task_res.task.created_at = task_res_created_at
            task_res.task.ttl = task_res_ttl

            # Execute
            res = state.store_task_res(task_res)

            # Assert
            if expected_store_result:
                assert res is not None
            else:
                assert res is None

    def test_get_task_ins_not_return_expired(self) -> None:
        """Test get_task_ins not to return expired tasks."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins.task.created_at = time.time() - 5
        task_ins.task.ttl = 5.0

        # Execute
        state.store_task_ins(task_ins=task_ins)

        # Assert
        with patch("time.time", side_effect=lambda: task_ins.task.created_at + 6.1):
            task_ins_list = state.get_task_ins(node_id=1, limit=None)
            assert len(task_ins_list) == 0

    def test_get_task_res_expired_task_ins(self) -> None:
        """Test get_task_res to return error TaskRes if its TaskIns has expired."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins.task.created_at = time.time() - 5
        task_ins.task.ttl = 5.1

        task_id = state.store_task_ins(task_ins=task_ins)

        task_res = create_task_res(
            producer_node_id=1,
            anonymous=False,
            ancestry=[str(task_id)],
            run_id=run_id,
        )
        task_res.task.ttl = 0.1
        _ = state.store_task_res(task_res=task_res)

        with patch("time.time", side_effect=lambda: task_ins.task.created_at + 6.1):
            # Execute
            assert task_id is not None
            task_res_list = state.get_task_res(task_ids={task_id})

            # Assert
            assert len(task_res_list) == 1
            assert task_res_list[0].task.HasField("error")
            assert state.num_task_ins() == 1
            assert state.num_task_res() == 0

    def test_get_task_res_returns_empty_for_missing_taskins(self) -> None:
        """Test that get_task_res returns an empty result when the corresponding TaskIns
        does not exist."""
        # Prepare
        state = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins_id = "5b0a3fc2-edba-4525-a89a-04b83420b7c8"

        task_res = create_task_res(
            producer_node_id=1,
            anonymous=False,
            ancestry=[str(task_ins_id)],
            run_id=run_id,
        )
        _ = state.store_task_res(task_res=task_res)

        # Execute
        task_res_list = state.get_task_res(task_ids={UUID(task_ins_id)})

        # Assert
        assert len(task_res_list) == 1
        assert task_res_list[0].task.HasField("error")
        assert state.num_task_ins() == state.num_task_res() == 0

    def test_get_task_res_return_if_not_expired(self) -> None:
        """Test get_task_res to return TaskRes if its TaskIns exists and is not
        expired."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )
        task_ins.task.created_at = time.time() - 5
        task_ins.task.ttl = 7.1

        task_id = state.store_task_ins(task_ins=task_ins)

        task_res = create_task_res(
            producer_node_id=node_id,
            anonymous=False,
            ancestry=[str(task_id)],
            run_id=run_id,
        )
        task_res.task.ttl = 0.1
        _ = state.store_task_res(task_res=task_res)

        with patch("time.time", side_effect=lambda: task_ins.task.created_at + 6.1):
            # Execute
            assert task_id is not None
            task_res_list = state.get_task_res(task_ids={task_id})

            # Assert
            assert len(task_res_list) != 0

    def test_store_task_res_fail_if_consumer_producer_id_mismatch(self) -> None:
        """Test store_task_res to fail if there is a mismatch between the
        consumer_node_id of taskIns and the producer_node_id of taskRes."""
        # Prepare
        state = self.state_factory()
        node_id = state.create_node(1e3)
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        task_ins = create_task_ins(
            consumer_node_id=node_id, anonymous=False, run_id=run_id
        )

        task_id = state.store_task_ins(task_ins=task_ins)

        task_res = create_task_res(
            # Different than consumer_node_id
            producer_node_id=100 if node_id != 100 else 101,
            anonymous=False,
            ancestry=[str(task_id)],
            run_id=run_id,
        )

        # Execute
        task_res_uuid = state.store_task_res(task_res=task_res)

        # Assert
        assert task_res_uuid is None

    def test_get_set_serverapp_context(self) -> None:
        """Test get and set serverapp context."""
        # Prepare
        state: LinkState = self.state_factory()
        context = Context(
            run_id=1,
            node_id=0,
            node_config={"mock": "mock"},
            state=RecordSet(),
            run_config={"test": "test"},
        )
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())

        # Execute
        init = state.get_serverapp_context(run_id)
        state.set_serverapp_context(run_id, context)
        retrieved_context = state.get_serverapp_context(run_id)

        # Assert
        assert init is None
        assert retrieved_context == context

    def test_set_context_invalid_run_id(self) -> None:
        """Test set_serverapp_context with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        context = Context(
            run_id=1,
            node_id=0,
            node_config={"mock": "mock"},
            state=RecordSet(),
            run_config={"test": "test"},
        )

        # Execute and assert
        with self.assertRaises(ValueError):
            state.set_serverapp_context(61016, context)  # Invalid run_id

    def test_add_serverapp_log_invalid_run_id(self) -> None:
        """Test adding serverapp log with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        invalid_run_id = 99999
        log_entry = "Invalid log entry"

        # Execute and assert
        with self.assertRaises(ValueError):
            state.add_serverapp_log(invalid_run_id, log_entry)

    def test_get_serverapp_log_invalid_run_id(self) -> None:
        """Test retrieving serverapp log with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        invalid_run_id = 99999

        # Execute and assert
        with self.assertRaises(ValueError):
            state.get_serverapp_log(invalid_run_id, after_timestamp=None)

    def test_add_and_get_serverapp_log(self) -> None:
        """Test adding and retrieving serverapp logs."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        timestamp = now().timestamp()

        # Execute
        state.add_serverapp_log(run_id, log_entry_1)
        state.add_serverapp_log(run_id, log_entry_2)
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest > timestamp
        assert log_entry_1 + log_entry_2 == retrieved_logs

    def test_get_serverapp_log_after_timestamp(self) -> None:
        """Test retrieving serverapp logs after a specific timestamp."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        state.add_serverapp_log(run_id, log_entry_1)
        timestamp = now().timestamp()
        state.add_serverapp_log(run_id, log_entry_2)

        # Execute
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest > timestamp
        assert log_entry_1 not in retrieved_logs
        assert log_entry_2 == retrieved_logs

    def test_get_serverapp_log_after_timestamp_no_logs(self) -> None:
        """Test retrieving serverapp logs after a specific timestamp but no logs are
        found."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(None, None, "9f86d08", {}, ConfigsRecord())
        log_entry = "Log entry"
        state.add_serverapp_log(run_id, log_entry)
        timestamp = now().timestamp()

        # Execute
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest == 0
        assert retrieved_logs == ""

    def test_create_run_with_and_without_federation_options(self) -> None:
        """Test that the recording and fetching of federation options works."""
        # Prepare
        state = self.state_factory()
        # A run w/ federation options
        fed_options = ConfigsRecord({"setting-a": 123, "setting-b": [4, 5, 6]})
        run_id = state.create_run(
            None,
            None,
            "fffffff",
            {"mock_key": "mock_value"},
            federation_options=fed_options,
        )
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))

        # Execute
        fed_options_fetched = state.get_federation_options(run_id=run_id)

        # Assert
        assert fed_options_fetched == fed_options

        # Generate a run_id that doesn't exist. Then check None is returned
        unique_int = next(num for num in range(0, 1) if num not in {run_id})
        assert state.get_federation_options(run_id=unique_int) is None


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
            recordset=ProtoRecordSet(parameters={}, metrics={}, configs={}),
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )
    task.task.pushed_at = time.time()
    return task


def create_task_res(
    producer_node_id: int,
    anonymous: bool,
    ancestry: list[str],
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
            recordset=ProtoRecordSet(parameters={}, metrics={}, configs={}),
            ttl=DEFAULT_TTL,
            created_at=time.time(),
        ),
    )
    task_res.task.pushed_at = time.time()
    return task_res


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> LinkState:
        """Return InMemoryState."""
        return InMemoryLinkState()


class SqliteInMemoryStateTest(StateTest, unittest.TestCase):
    """Test SqliteState implemenation with in-memory database."""

    __test__ = True

    def state_factory(self) -> SqliteLinkState:
        """Return SqliteState with in-memory database."""
        state = SqliteLinkState(":memory:")
        state.initialize()
        return state

    def test_initialize(self) -> None:
        """Test initialization."""
        # Prepare
        state = self.state_factory()

        # Execute
        result = state.query("SELECT name FROM sqlite_schema;")

        # Assert
        assert len(result) == 17


class SqliteFileBasedTest(StateTest, unittest.TestCase):
    """Test SqliteState implemenation with file-based database."""

    __test__ = True

    def state_factory(self) -> SqliteLinkState:
        """Return SqliteState with file-based database."""
        # pylint: disable-next=consider-using-with,attribute-defined-outside-init
        self.tmp_file = tempfile.NamedTemporaryFile()
        state = SqliteLinkState(database_path=self.tmp_file.name)
        state.initialize()
        return state

    def test_initialize(self) -> None:
        """Test initialization."""
        # Prepare
        state = self.state_factory()

        # Execute
        result = state.query("SELECT name FROM sqlite_schema;")

        # Assert
        assert len(result) == 17


if __name__ == "__main__":
    unittest.main(verbosity=2)

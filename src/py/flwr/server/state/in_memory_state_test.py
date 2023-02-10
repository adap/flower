# Copyright 2022 Adap GmbH. All Rights Reserved.
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
"""InMemortState tests."""


from datetime import datetime, timezone
from uuid import uuid4

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns

from .in_memory_state import InMemoryState


def test_get_task_ins_empty() -> None:
    """Validate that a new state has no TaskIns."""

    # Prepare
    state = InMemoryState()

    # Execute
    task_ins_list = state.get_task_ins(
        node_id=1,
        limit=10,
    )

    # Assert
    assert not task_ins_list


def test_get_task_ins_identity() -> None:
    """Validate that a new state has no TaskIns."""

    # Prepare
    state = InMemoryState()
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


def test_get_task_ins_anonymous() -> None:
    """Validate that a new state has no TaskIns."""

    # Prepare
    state = InMemoryState()
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


def test_get_task_res_empty() -> None:
    """Validate that a new state has no TaskRes."""

    # Prepare
    state = InMemoryState()

    # Execute
    task_res_list = state.get_task_res(
        task_ids={uuid4()},
        limit=10,
    )

    # Assert
    assert not task_res_list


def test_store_task_ins_one() -> None:
    """Test store_task_ins."""

    # Prepare
    node_id = 1
    state = InMemoryState()
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

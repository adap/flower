# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for module task_handler."""


from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import RecordSet, serde
from flwr.proto.fleet_pb2 import PullTaskInsResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns  # pylint: disable=E0611


def test_validate_task_ins_no_task() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=None)

    assert not validate_task_ins(task_ins)


def test_validate_task_ins_no_content() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=Task(recordset=None))

    assert not validate_task_ins(task_ins)


def test_validate_task_ins_valid() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=Task(recordset=serde.recordset_to_proto(RecordSet())))

    assert validate_task_ins(task_ins)


def test_get_task_ins_empty_response() -> None:
    """Test get_task_ins."""
    res = PullTaskInsResponse(reconnect=None, task_ins_list=[])
    task_ins = get_task_ins(res)
    assert task_ins is None


def test_get_task_ins_single_ins() -> None:
    """Test get_task_ins."""
    expected_task_ins = TaskIns(task_id="123", task=Task())
    res = PullTaskInsResponse(reconnect=None, task_ins_list=[expected_task_ins])
    actual_task_ins = get_task_ins(res)
    assert actual_task_ins == expected_task_ins


def test_get_task_ins_multiple_ins() -> None:
    """Test get_task_ins."""
    expected_task_ins = TaskIns(task_id="123", task=Task())
    res = PullTaskInsResponse(
        reconnect=None, task_ins_list=[expected_task_ins, TaskIns(), TaskIns()]
    )
    actual_task_ins = get_task_ins(res)
    assert actual_task_ins == expected_task_ins
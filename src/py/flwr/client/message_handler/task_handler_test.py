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


from flwr.client.message_handler.task_handler import validate_task_ins
from flwr.common import RecordSet, serde
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

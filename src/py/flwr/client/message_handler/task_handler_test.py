# Copyright 2023 Adap GmbH. All Rights Reserved.
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


from flwr.client.message_handler.task_handler import get_server_message
from flwr.proto.fleet_pb2 import PullTaskInsResponse, Reconnect
from flwr.proto.task_pb2 import Task, TaskIns
from flwr.proto.transport_pb2 import ServerMessage


def test_get_server_message_empty() -> None:
    """Test get_server_message."""

    # Prepare
    res = PullTaskInsResponse(reconnect=None, task_ins_list=[])

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is None


def test_get_server_message_reconnect() -> None:
    """Test get_server_message."""

    # Prepare
    res = PullTaskInsResponse(reconnect=Reconnect(reconnect=42), task_ins_list=[])

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is None


def test_get_server_message_none_task() -> None:
    """Test get_server_message."""

    # Prepare
    res = PullTaskInsResponse(reconnect=None, task_ins_list=[TaskIns(task=None)])

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is None


def test_get_server_message_none_legacy() -> None:
    """Test get_server_message."""

    # Prepare
    res = PullTaskInsResponse(
        reconnect=None, task_ins_list=[TaskIns(task=Task(legacy_server_message=None))]
    )

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is None


def test_get_server_message_legacy_reconnect() -> None:
    """Test get_server_message."""

    # Prepare
    res = PullTaskInsResponse(
        reconnect=None,
        task_ins_list=[
            TaskIns(
                task=Task(
                    legacy_server_message=ServerMessage(
                        reconnect_ins=ServerMessage.ReconnectIns(seconds=3)
                    )
                )
            )
        ],
    )

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is None


def test_get_server_message_legacy_valid() -> None:
    """Test get_server_message."""

    # Prepare
    expected = TaskIns(
        task=Task(
            legacy_server_message=ServerMessage(
                get_properties_ins=ServerMessage.GetPropertiesIns()
            )
        )
    )
    res = PullTaskInsResponse(
        reconnect=None,
        task_ins_list=[expected],
    )

    # Execute
    actual = get_server_message(res)

    # Assert
    assert actual is not None
    actual_task_ins, actual_server_message = actual
    assert actual_task_ins == expected

    # pylint: disable=no-member
    assert actual_server_message == expected.task.legacy_server_message
    # pylint: enable=no-member

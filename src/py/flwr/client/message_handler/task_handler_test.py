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


from flwr.client.message_handler.task_handler import (
    get_server_message_from_task_ins,
    get_task_ins,
    validate_task_ins,
    validate_task_res,
    wrap_client_message_in_task_res,
)
from flwr.proto.fleet_pb2 import PullTaskInsResponse
from flwr.proto.task_pb2 import SecureAggregation, Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


def test_validate_task_ins_no_task() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=None)

    assert not validate_task_ins(task_ins, discard_reconnect_ins=True)
    assert not validate_task_ins(task_ins, discard_reconnect_ins=False)


def test_validate_task_ins_no_content() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=Task(legacy_server_message=None, sa=None))

    assert not validate_task_ins(task_ins, discard_reconnect_ins=True)
    assert not validate_task_ins(task_ins, discard_reconnect_ins=False)


def test_validate_task_ins_with_reconnect_ins() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(
        task=Task(
            legacy_server_message=ServerMessage(
                reconnect_ins=ServerMessage.ReconnectIns(seconds=3)
            )
        )
    )

    assert not validate_task_ins(task_ins, discard_reconnect_ins=True)
    assert validate_task_ins(task_ins, discard_reconnect_ins=False)


def test_validate_task_ins_valid_legacy_server_message() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(
        task=Task(
            legacy_server_message=ServerMessage(
                get_properties_ins=ServerMessage.GetPropertiesIns()
            )
        )
    )

    assert validate_task_ins(task_ins, discard_reconnect_ins=True)
    assert validate_task_ins(task_ins, discard_reconnect_ins=False)


def test_validate_task_ins_valid_sa() -> None:
    """Test validate_task_ins."""
    task_ins = TaskIns(task=Task(sa=SecureAggregation()))

    assert validate_task_ins(task_ins, discard_reconnect_ins=True)
    assert validate_task_ins(task_ins, discard_reconnect_ins=False)


def test_validate_task_res() -> None:
    """Test validate_task_res."""
    task_res = TaskRes(task=Task())
    assert validate_task_res(task_res)

    task_res.task_id = "123"
    assert not validate_task_res(task_res)

    task_res.Clear()
    task_res.group_id = "123"
    assert not validate_task_res(task_res)

    task_res.Clear()
    task_res.workload_id = "123"
    assert not validate_task_res(task_res)

    task_res.Clear()
    # pylint: disable-next=no-member
    task_res.task.producer.node_id = 0
    assert not validate_task_res(task_res)

    task_res.Clear()
    # pylint: disable-next=no-member
    task_res.task.consumer.node_id = 0
    assert not validate_task_res(task_res)

    task_res.Clear()
    # pylint: disable-next=no-member
    task_res.task.ancestry.append("123")
    assert not validate_task_res(task_res)


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


def test_get_server_message_from_task_ins_invalid() -> None:
    """Test get_server_message_from_task_ins."""
    task_ins = TaskIns(task=Task(legacy_server_message=None))
    msg_t = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=True)
    msg_f = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)

    assert msg_t is None
    assert msg_f is None


def test_get_server_message_from_task_ins_reconnect_ins() -> None:
    """Test get_server_message_from_task_ins."""
    expected_server_message = ServerMessage(
        reconnect_ins=ServerMessage.ReconnectIns(seconds=3)
    )
    task_ins = TaskIns(task=Task(legacy_server_message=expected_server_message))
    msg_t = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=True)
    msg_f = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)

    assert msg_t is None
    assert msg_f == expected_server_message


def test_get_server_message_from_task_ins_sa() -> None:
    """Test get_server_message_from_task_ins."""
    task_ins = TaskIns(task=Task(sa=SecureAggregation()))
    msg_t = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=True)
    msg_f = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)

    assert msg_t is None
    assert msg_f is None


def test_get_server_message_from_task_ins_valid_legacy_server_message() -> None:
    """Test get_server_message_from_task_ins."""
    expected_server_message = ServerMessage(
        get_properties_ins=ServerMessage.GetPropertiesIns()
    )
    task_ins = TaskIns(task=Task(legacy_server_message=expected_server_message))
    msg_t = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=True)
    msg_f = get_server_message_from_task_ins(task_ins, exclude_reconnect_ins=False)

    assert msg_t == expected_server_message
    assert msg_f == expected_server_message


def test_wrap_client_message_in_task_res() -> None:
    """Test wrap_client_message_in_task_res."""
    expected_client_message = ClientMessage(
        get_properties_res=ClientMessage.GetPropertiesRes()
    )
    task_res = wrap_client_message_in_task_res(expected_client_message)

    assert validate_task_res(task_res)
    # pylint: disable-next=no-member
    assert task_res.task.legacy_client_message == expected_client_message

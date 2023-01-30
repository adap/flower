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
"""DriverServicer tests."""


import uuid

from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns
from flwr.proto.transport_pb2 import ServerMessage
from flwr.server.driver.driver_servicer import _raise_if, _validate_incoming_task_ins

VALUE_ERROR_BASE: str = "Malformed PushTaskInsRequest: "

# pylint: disable=broad-except


def test_raise_if_false() -> None:
    """."""

    # Prepare
    validation_error = False
    detail = "test"

    try:
        # Execute
        _raise_if(validation_error, detail)

        # Assert
        assert True
    except ValueError:
        assert False
    except Exception:
        assert False


def test_raise_if_true() -> None:
    """."""

    # Prepare
    validation_error = True
    detail = "test"

    try:
        # Execute
        _raise_if(validation_error, detail)

        # Assert
        assert False
    except ValueError as err:
        assert str(err) == "Malformed PushTaskInsRequest: test"
    except Exception:
        assert False


def _create_task_ins(
    task_id: str = "", task: bool = True, server_message: bool = True
) -> TaskIns:
    return TaskIns(
        task_id=task_id,
        group_id="",
        workload_id="",
        task=Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=1, anonymous=False),
            created_at="",
            delivered_at="",
            ttl="",
            ancestry=[],
            legacy_server_message=ServerMessage(fit_ins=ServerMessage.FitIns())
            if server_message
            else None,
            legacy_client_message=None,
        )
        if task
        else None,
    )


def test_validate_incoming_task_ins_valid() -> None:
    """Test TaskIns validation."""

    # Prepare
    task_ins = _create_task_ins()

    # Execute
    try:
        _validate_incoming_task_ins(task_ins=task_ins)

        # Assert
        assert True
    except Exception:
        assert False


def test_validate_incoming_task_ins_invalid_task_id_set() -> None:
    """Test TaskIns validation."""

    # Prepare
    task_ins = _create_task_ins(task_id=str(uuid.uuid4()))

    # Execute
    try:
        _validate_incoming_task_ins(task_ins=task_ins)

        # Assert
        assert False
    except ValueError as err:
        assert str(err).startswith(VALUE_ERROR_BASE)
    except Exception:
        assert False


def test_validate_incoming_task_ins_invalid_no_task() -> None:
    """Test TaskIns validation."""

    # Prepare
    task_ins = _create_task_ins(task=False)

    # Execute
    try:
        _validate_incoming_task_ins(task_ins=task_ins)

        # Assert
        assert False
    except ValueError as err:
        assert str(err).startswith(VALUE_ERROR_BASE)
    except Exception:
        assert False


def test_validate_incoming_task_ins_invalid_no_server_message() -> None:
    """Test TaskIns validation."""

    # Prepare
    task_ins = _create_task_ins(server_message=False)

    # Execute
    try:
        _validate_incoming_task_ins(task_ins=task_ins)

        # Assert
        assert False
    except ValueError as err:
        assert str(err).startswith(VALUE_ERROR_BASE)
    except Exception:
        assert False

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
"""Task handling."""


from typing import Optional

from flwr.proto.fleet_pb2 import PullTaskInsResponse
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


def validate_task_ins(task_ins: TaskIns, discard_reconnect_ins: bool) -> bool:
    """Validate a TaskIns before it entering the message handling process.

    Parameters
    ----------
    task_ins: TaskIns
        The task instruction coming from the server.
    discard_reconnect_ins: bool
        If True, ReconnectIns will not be considered as valid content.

    Returns
    -------
    is_valid: bool
        True if the TaskIns is deemed valid and therefore suitable for
        undergoing the message handling process, False otherwise.
    """
    # Check if the task_ins contains legacy_server_message.
    # If so, check if ServerMessage is one of
    # {GetPropertiesIns, GetParametersIns, FitIns, EvaluateIns, ReconnectIns*}
    # Discard ReconnectIns if discard_reconnect_ins is true.
    if (
        not task_ins.HasField("task")
        or not task_ins.task.HasField("legacy_server_message")
        or (
            discard_reconnect_ins
            and task_ins.task.legacy_server_message.WhichOneof("msg") == "reconnect_ins"
        )
    ):
        return False

    return True


def get_task_ins(
    pull_task_ins_response: PullTaskInsResponse,
) -> Optional[TaskIns]:
    """Get the first TaskIns, if available."""
    # Extract a single ServerMessage from the response, if possible
    if len(pull_task_ins_response.task_ins_list) == 0:
        return None

    # Only evaluate the first message
    task_ins: TaskIns = pull_task_ins_response.task_ins_list[0]

    return task_ins


def get_server_message_from_task_ins(
    task_ins: TaskIns, exclude_reconnect_ins: bool
) -> Optional[ServerMessage]:
    """Get ServerMessage from TaskIns, if available."""
    # Return the message if it is in
    # {GetPropertiesIns, GetParametersIns, FitIns, EvaluateIns}
    # Return the message if it is ReconnectIns and exclude_reconnect_ins is False.
    if not validate_task_ins(task_ins, discard_reconnect_ins=exclude_reconnect_ins):
        return None

    return task_ins.task.legacy_server_message


def wrap_client_message_in_task_res(client_message: ClientMessage) -> TaskRes:
    """Wrap ClientMessage in TaskRes."""
    # Instantiate a TaskRes, only filling client_message field.
    return TaskRes(
        task_id="",
        group_id="",
        workload_id="",
        task=Task(ancestry=[], legacy_client_message=client_message),
    )

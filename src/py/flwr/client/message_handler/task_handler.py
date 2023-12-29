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
"""Task handling."""


from typing import Optional

from flwr.proto.fleet_pb2 import PullTaskInsResponse
from flwr.proto.node_pb2 import Node
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
    # Check if the task_ins contains legacy_server_message or sa.
    # If legacy_server_message is set, check if ServerMessage is one of
    # {GetPropertiesIns, GetParametersIns, FitIns, EvaluateIns, ReconnectIns*}
    # Discard ReconnectIns if discard_reconnect_ins is true.
    if (
        not task_ins.HasField("task")
        or (
            not task_ins.task.HasField("legacy_server_message")
            and not task_ins.task.HasField("sa")
        )
        or (
            discard_reconnect_ins
            and task_ins.task.legacy_server_message.WhichOneof("msg") == "reconnect_ins"
        )
    ):
        return False

    return True


def validate_task_res(task_res: TaskRes) -> bool:
    """Validate a TaskRes before filling its fields in the `send()` function.

    Parameters
    ----------
    task_res: TaskRes
        The task response to be sent to the server.

    Returns
    -------
    is_valid: bool
        True if the `task_id`, `group_id`, and `workload_id` fields in TaskRes
        and the `producer`, `consumer`, and `ancestry` fields in its sub-message Task
        are not initialized accidentally elsewhere,
        False otherwise.
    """
    # Retrieve initialized fields in TaskRes and Task
    initialized_fields_in_task_res = {field.name for field, _ in task_res.ListFields()}
    initialized_fields_in_task = {field.name for field, _ in task_res.task.ListFields()}

    # Check if certain fields are already initialized
    # pylint: disable-next=too-many-boolean-expressions
    if (
        "task_id" in initialized_fields_in_task_res
        or "group_id" in initialized_fields_in_task_res
        or "workload_id" in initialized_fields_in_task_res
        or "producer" in initialized_fields_in_task
        or "consumer" in initialized_fields_in_task
        or "ancestry" in initialized_fields_in_task
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
    if not validate_task_ins(
        task_ins, discard_reconnect_ins=exclude_reconnect_ins
    ) or not task_ins.task.HasField("legacy_server_message"):
        return None

    return task_ins.task.legacy_server_message


def wrap_client_message_in_task_res(client_message: ClientMessage) -> TaskRes:
    """Wrap ClientMessage in TaskRes."""
    # Instantiate a TaskRes, only filling client_message field.
    return TaskRes(
        task_id="",
        group_id="",
        workload_id=0,
        task=Task(ancestry=[], legacy_client_message=client_message),
    )


def configure_task_res(
    task_res: TaskRes, ref_task_ins: TaskIns, producer: Node
) -> TaskRes:
    """Set the metadata of a TaskRes.

    Fill `group_id` and `workload_id` in TaskRes
    and `producer`, `consumer`, and `ancestry` in Task in TaskRes.

    `producer` in Task in TaskRes will remain unchanged/unset.

    Note that protobuf API `protobuf.message.MergeFrom(other_msg)`
    does NOT always overwrite fields that are set in `other_msg`.
    Please refer to:
    https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html
    """
    task_res = TaskRes(
        task_id="",  # This will be generated by the server
        group_id=ref_task_ins.group_id,
        workload_id=ref_task_ins.workload_id,
        task=task_res.task,
    )
    # pylint: disable-next=no-member
    task_res.task.MergeFrom(
        Task(
            producer=producer,
            consumer=ref_task_ins.task.producer,
            ancestry=[ref_task_ins.task_id],
        )
    )
    return task_res

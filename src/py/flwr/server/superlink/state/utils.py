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
"""Utility functions for State."""


from logging import ERROR
from os import urandom
from uuid import uuid4

from typing import Union
from flwr.common import log, now
from flwr.common.constant import ErrorCode
from flwr.proto.error_pb2 import Error  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611

NODE_UNAVAILABLE_ERROR_REASON = (
    "Error: Node Unavailable - The destination node is currently unavailable. "
    "It exceeds the time limit specified in its last ping."
)
MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Message Unavailable - The requested message could not be found in the "
    "database. It may have expired due to its TTL or never existed."
)
REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Reply Message Unavailable - The reply message has expired."
)


def generate_rand_int_from_bytes(num_bytes: int) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes."""
    return int.from_bytes(urandom(num_bytes), "little", signed=False)


def convert_uint64_to_sint64(u: int) -> int:
    """Convert a uint64 value to a sint64 value with the same bit sequence.

    Parameters
    ----------
    u : int
        The unsigned 64-bit integer to convert.

    Returns
    -------
    int
        The signed 64-bit integer equivalent.

        The signed 64-bit integer will have the same bit pattern as the
        unsigned 64-bit integer but may have a different decimal value.

        For numbers within the range [0, `sint64` max value], the decimal
        value remains the same. However, for numbers greater than the `sint64`
        max value, the decimal value will differ due to the wraparound caused
        by the sign bit.
    """
    if u >= (1 << 63):
        return u - (1 << 64)
    return u


def convert_sint64_to_uint64(s: int) -> int:
    """Convert a sint64 value to a uint64 value with the same bit sequence.

    Parameters
    ----------
    s : int
        The signed 64-bit integer to convert.

    Returns
    -------
    int
        The unsigned 64-bit integer equivalent.

        The unsigned 64-bit integer will have the same bit pattern as the
        signed 64-bit integer but may have a different decimal value.

        For negative `sint64` values, the conversion adds 2^64 to the
        signed value to obtain the equivalent `uint64` value. For non-negative
        `sint64` values, the decimal value remains unchanged in the `uint64`
        representation.
    """
    if s < 0:
        return s + (1 << 64)
    return s


def convert_uint64_values_in_dict_to_sint64(
    data_dict: dict[str, int], keys: list[str]
) -> None:
    """Convert uint64 values to sint64 in the given dictionary.

    Parameters
    ----------
    data_dict : dict[str, int]
        A dictionary where the values are integers to be converted.
    keys : list[str]
        A list of keys in the dictionary whose values need to be converted.
    """
    for key in keys:
        if key in data_dict:
            data_dict[key] = convert_uint64_to_sint64(data_dict[key])


def convert_sint64_values_in_dict_to_uint64(
    data_dict: dict[str, int], keys: list[str]
) -> None:
    """Convert sint64 values to uint64 in the given dictionary.

    Parameters
    ----------
    data_dict : dict[str, int]
        A dictionary where the values are integers to be converted.
    keys : list[str]
        A list of keys in the dictionary whose values need to be converted.
    """
    for key in keys:
        if key in data_dict:
            data_dict[key] = convert_sint64_to_uint64(data_dict[key])


def make_node_unavailable_taskres(ref_taskins: TaskIns) -> TaskRes:
    """Generate a TaskRes with a node unavailable error from a TaskIns."""
    current_time = now().timestamp()
    ttl = ref_taskins.task.ttl - (current_time - ref_taskins.task.created_at)
    if ttl < 0:
        log(ERROR, "Creating TaskRes for TaskIns that exceeds its TTL.")
        ttl = 0
    return TaskRes(
        task_id=str(uuid4()),
        group_id=ref_taskins.group_id,
        run_id=ref_taskins.run_id,
        task=Task(
            producer=Node(node_id=ref_taskins.task.consumer.node_id, anonymous=False),
            consumer=Node(node_id=ref_taskins.task.producer.node_id, anonymous=False),
            created_at=current_time,
            ttl=ttl,
            ancestry=[ref_taskins.task_id],
            task_type=ref_taskins.task.task_type,
            error=Error(
                code=ErrorCode.NODE_UNAVAILABLE, reason=NODE_UNAVAILABLE_ERROR_REASON
            ),
        ),
    )


def make_taskins_unavailable_taskres(taskins_id: str) -> TaskRes:
    """Generate a TaskRes with a taskins unavailable error."""
    current_time = now().timestamp()
    return TaskRes(
        task_id=str(uuid4()),
        group_id="",  # Unknown group ID
        run_id=0,  # Unknown run ID
        task=Task(
            # This function is only called by SuperLink, and thus it's the producer.
            producer=Node(node_id=0, anonymous=False),
            consumer=Node(node_id=0, anonymous=False),
            created_at=current_time,
            ttl=0,
            ancestry=[taskins_id],
            task_type="",  # Unknown message type
            error=Error(
                code=ErrorCode.MESSAGE_UNAVAILABLE,
                reason=MESSAGE_UNAVAILABLE_ERROR_REASON,
            ),
        ),
    )


def make_taskres_unavailable_taskres(ref_taskins: TaskIns) -> TaskRes:
    """Generate a TaskRes with a reply message unavailable error from a TaskIns."""
    current_time = now().timestamp()
    ttl = ref_taskins.task.ttl - (current_time - ref_taskins.task.created_at)
    if ttl < 0:
        log(ERROR, "Creating TaskRes for TaskIns that exceeds its TTL.")
        ttl = 0
    return TaskRes(
        task_id=str(uuid4()),
        group_id=ref_taskins.group_id,
        run_id=ref_taskins.run_id,
        task=Task(
            # This function is only called by SuperLink, and thus it's the producer.
            producer=Node(node_id=0, anonymous=False),
            consumer=Node(node_id=0, anonymous=False),
            created_at=current_time,
            ttl=ttl,
            ancestry=[ref_taskins.task_id],
            task_type=ref_taskins.task.task_type,
            error=Error(
                code=ErrorCode.REPLY_MESSAGE_UNAVAILABLE,
                reason=REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON,
            ),
        ),
    )


def has_expired(task_ins_or_res: Union[TaskIns, TaskRes], current_time: float) -> bool:
    """Check if the TaskIns/TaskRes has expired."""
    return task_ins_or_res.task.ttl + task_ins_or_res.task.created_at < current_time

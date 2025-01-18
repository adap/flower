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
from typing import Optional, Union
from uuid import UUID, uuid4

from flwr.common import ConfigsRecord, Context, log, now, serde
from flwr.common.constant import SUPERLINK_NODE_ID, ErrorCode, Status, SubStatus
from flwr.common.typing import RunStatus

# pylint: disable=E0611
from flwr.proto.error_pb2 import Error
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.node_pb2 import Node
from flwr.proto.recordset_pb2 import ConfigsRecord as ProtoConfigsRecord
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes

# pylint: enable=E0611

NODE_UNAVAILABLE_ERROR_REASON = (
    "Error: Node Unavailable - The destination node is currently unavailable. "
    "It exceeds the time limit specified in its last ping."
)

VALID_RUN_STATUS_TRANSITIONS = {
    (Status.PENDING, Status.STARTING),
    (Status.STARTING, Status.RUNNING),
    (Status.RUNNING, Status.FINISHED),
    # Any non-FINISHED status can transition to FINISHED
    (Status.PENDING, Status.FINISHED),
    (Status.STARTING, Status.FINISHED),
}
VALID_RUN_SUB_STATUSES = {
    SubStatus.COMPLETED,
    SubStatus.FAILED,
    SubStatus.STOPPED,
}
MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Message Unavailable - The requested message could not be found in the "
    "database. It may have expired due to its TTL or never existed."
)
REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Reply Message Unavailable - The reply message has expired."
)


def generate_rand_int_from_bytes(
    num_bytes: int, exclude: Optional[list[int]] = None
) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes.

    If `exclude` is set, this function guarantees such number is not returned.
    """
    num = int.from_bytes(urandom(num_bytes), "little", signed=False)

    if exclude:
        while num in exclude:
            num = int.from_bytes(urandom(num_bytes), "little", signed=False)
    return num


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


def context_to_bytes(context: Context) -> bytes:
    """Serialize `Context` to bytes."""
    return serde.context_to_proto(context).SerializeToString()


def context_from_bytes(context_bytes: bytes) -> Context:
    """Deserialize `Context` from bytes."""
    return serde.context_from_proto(ProtoContext.FromString(context_bytes))


def configsrecord_to_bytes(configs_record: ConfigsRecord) -> bytes:
    """Serialize a `ConfigsRecord` to bytes."""
    return serde.configs_record_to_proto(configs_record).SerializeToString()


def configsrecord_from_bytes(configsrecord_bytes: bytes) -> ConfigsRecord:
    """Deserialize `ConfigsRecord` from bytes."""
    return serde.configs_record_from_proto(
        ProtoConfigsRecord.FromString(configsrecord_bytes)
    )


def is_valid_transition(current_status: RunStatus, new_status: RunStatus) -> bool:
    """Check if a transition between two run statuses is valid.

    Parameters
    ----------
    current_status : RunStatus
        The current status of the run.
    new_status : RunStatus
        The new status to transition to.

    Returns
    -------
    bool
        True if the transition is valid, False otherwise.
    """
    # Transition to FINISHED from a non-RUNNING status is only allowed
    # if the sub-status is not COMPLETED
    if (
        current_status.status in [Status.PENDING, Status.STARTING]
        and new_status.status == Status.FINISHED
    ):
        return new_status.sub_status != SubStatus.COMPLETED

    return (
        current_status.status,
        new_status.status,
    ) in VALID_RUN_STATUS_TRANSITIONS


def has_valid_sub_status(status: RunStatus) -> bool:
    """Check if the 'sub_status' field of the given status is valid.

    Parameters
    ----------
    status : RunStatus
        The status object to be checked.

    Returns
    -------
    bool
        True if the status object has a valid sub-status, False otherwise.

    Notes
    -----
    Only an empty string (i.e., "") is considered a valid sub-status for
    non-finished statuses. The sub-status of a finished status cannot be empty.
    """
    if status.status == Status.FINISHED:
        return status.sub_status in VALID_RUN_SUB_STATUSES
    return status.sub_status == ""


def create_taskres_for_unavailable_taskins(taskins_id: Union[str, UUID]) -> TaskRes:
    """Generate a TaskRes with a TaskIns unavailable error.

    Parameters
    ----------
    taskins_id : Union[str, UUID]
        The ID of the unavailable TaskIns.

    Returns
    -------
    TaskRes
        A TaskRes with an error code MESSAGE_UNAVAILABLE to indicate that the
        inquired TaskIns ID cannot be found (due to non-existence or expiration).
    """
    current_time = now().timestamp()
    return TaskRes(
        task_id=str(uuid4()),
        group_id="",  # Unknown group ID
        run_id=0,  # Unknown run ID
        task=Task(
            # This function is only called by SuperLink, and thus it's the producer.
            producer=Node(node_id=SUPERLINK_NODE_ID),
            consumer=Node(node_id=SUPERLINK_NODE_ID),
            created_at=current_time,
            ttl=0,
            ancestry=[str(taskins_id)],
            task_type="",  # Unknown message type
            error=Error(
                code=ErrorCode.MESSAGE_UNAVAILABLE,
                reason=MESSAGE_UNAVAILABLE_ERROR_REASON,
            ),
        ),
    )


def create_taskres_for_unavailable_taskres(ref_taskins: TaskIns) -> TaskRes:
    """Generate a TaskRes with a reply message unavailable error from a TaskIns.

    Parameters
    ----------
    ref_taskins : TaskIns
        The reference TaskIns object.

    Returns
    -------
    TaskRes
        The generated TaskRes with an error code REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON,
        indicating that the original TaskRes has expired.
    """
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
            producer=Node(node_id=SUPERLINK_NODE_ID),
            consumer=Node(node_id=SUPERLINK_NODE_ID),
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


def verify_taskins_ids(
    inquired_taskins_ids: set[UUID],
    found_taskins_dict: dict[UUID, TaskIns],
    current_time: Optional[float] = None,
    update_set: bool = True,
) -> dict[UUID, TaskRes]:
    """Verify found TaskIns and generate error TaskRes for invalid ones.

    Parameters
    ----------
    inquired_taskins_ids : set[UUID]
        Set of TaskIns IDs for which to generate error TaskRes if invalid.
    found_taskins_dict : dict[UUID, TaskIns]
        Dictionary containing all found TaskIns indexed by their IDs.
    current_time : Optional[float] (default: None)
        The current time to check for expiration. If set to `None`, the current time
        will automatically be set to the current timestamp using `now().timestamp()`.
    update_set : bool (default: True)
        If True, the `inquired_taskins_ids` will be updated to remove invalid ones,
        by default True.

    Returns
    -------
    dict[UUID, TaskRes]
        A dictionary of error TaskRes indexed by the corresponding TaskIns ID.
    """
    ret_dict = {}
    current = current_time if current_time else now().timestamp()
    for taskins_id in list(inquired_taskins_ids):
        # Generate error TaskRes if the task_ins doesn't exist or has expired
        taskins = found_taskins_dict.get(taskins_id)
        if taskins is None or has_expired(taskins, current):
            if update_set:
                inquired_taskins_ids.remove(taskins_id)
            taskres = create_taskres_for_unavailable_taskins(taskins_id)
            ret_dict[taskins_id] = taskres
    return ret_dict


def verify_found_taskres(
    inquired_taskins_ids: set[UUID],
    found_taskins_dict: dict[UUID, TaskIns],
    found_taskres_list: list[TaskRes],
    current_time: Optional[float] = None,
    update_set: bool = True,
) -> dict[UUID, TaskRes]:
    """Verify found TaskRes and generate error TaskRes for invalid ones.

    Parameters
    ----------
    inquired_taskins_ids : set[UUID]
        Set of TaskIns IDs for which to generate error TaskRes if invalid.
    found_taskins_dict : dict[UUID, TaskIns]
        Dictionary containing all found TaskIns indexed by their IDs.
    found_taskres_list : dict[TaskIns, TaskRes]
        List of found TaskRes to be verified.
    current_time : Optional[float] (default: None)
        The current time to check for expiration. If set to `None`, the current time
        will automatically be set to the current timestamp using `now().timestamp()`.
    update_set : bool (default: True)
        If True, the `inquired_taskins_ids` will be updated to remove ones
        that have a TaskRes, by default True.

    Returns
    -------
    dict[UUID, TaskRes]
        A dictionary of TaskRes indexed by the corresponding TaskIns ID.
    """
    ret_dict: dict[UUID, TaskRes] = {}
    current = current_time if current_time else now().timestamp()
    for taskres in found_taskres_list:
        taskins_id = UUID(taskres.task.ancestry[0])
        if update_set:
            inquired_taskins_ids.remove(taskins_id)
        # Check if the TaskRes has expired
        if has_expired(taskres, current):
            # No need to insert the error TaskRes
            taskres = create_taskres_for_unavailable_taskres(
                found_taskins_dict[taskins_id]
            )
            taskres.task.delivered_at = now().isoformat()
        ret_dict[taskins_id] = taskres
    return ret_dict

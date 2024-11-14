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


from os import urandom

from flwr.common import ConfigsRecord, Context, serde
from flwr.common.constant import Status, SubStatus
from flwr.common.typing import RunStatus
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611

# pylint: disable=E0611
from flwr.proto.recordset_pb2 import ConfigsRecord as ProtoConfigsRecord

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

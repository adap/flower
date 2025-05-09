# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utils for serde."""

from collections.abc import MutableMapping
from typing import Any, TypeVar, cast

from google.protobuf.message import Message as GrpcMessage

# pylint: disable=E0611
from flwr.proto.recorddict_pb2 import (
    BoolList,
    BytesList,
    DoubleList,
    SintList,
    StringList,
    UintList,
)

from .constant import INT64_MAX_VALUE
from .record.typeddict import TypedDict

_type_to_field: dict[type, str] = {
    float: "double",
    int: "sint64",
    bool: "bool",
    str: "string",
    bytes: "bytes",
}
_list_type_to_class_and_field: dict[type, tuple[type[GrpcMessage], str]] = {
    float: (DoubleList, "double_list"),
    int: (SintList, "sint_list"),
    bool: (BoolList, "bool_list"),
    str: (StringList, "string_list"),
    bytes: (BytesList, "bytes_list"),
}
T = TypeVar("T")


def _is_uint64(value: Any) -> bool:
    """Check if a value is uint64."""
    return isinstance(value, int) and value > INT64_MAX_VALUE


def _record_value_to_proto(
    value: Any, allowed_types: list[type], proto_class: type[T]
) -> T:
    """Serialize `*RecordValue` to ProtoBuf.

    Note: `bool` MUST be put in the front of allowd_types if it exists.
    """
    arg = {}
    for t in allowed_types:
        # Single element
        # Note: `isinstance(False, int) == True`.
        if isinstance(value, t):
            fld = _type_to_field[t]
            if t is int and _is_uint64(value):
                fld = "uint64"
            arg[fld] = value
            return proto_class(**arg)
        # List
        if isinstance(value, list) and all(isinstance(item, t) for item in value):
            list_class, fld = _list_type_to_class_and_field[t]
            # Use UintList if any element is of type `uint64`.
            if t is int and any(_is_uint64(v) for v in value):
                list_class, fld = UintList, "uint_list"
            arg[fld] = list_class(vals=value)
            return proto_class(**arg)
    # Invalid types
    raise TypeError(
        f"The type of the following value is not allowed "
        f"in '{proto_class.__name__}':\n{value}"
    )


def _record_value_from_proto(value_proto: GrpcMessage) -> Any:
    """Deserialize `*RecordValue` from ProtoBuf."""
    value_field = cast(str, value_proto.WhichOneof("value"))
    if value_field.endswith("list"):
        value = list(getattr(value_proto, value_field).vals)
    else:
        value = getattr(value_proto, value_field)
    return value


def record_value_dict_to_proto(
    value_dict: TypedDict[str, Any],
    allowed_types: list[type],
    value_proto_class: type[T],
) -> dict[str, T]:
    """Serialize the record value dict to ProtoBuf.

    Note: `bool` MUST be put in the front of allowd_types if it exists.
    """
    # Move bool to the front
    if bool in allowed_types and allowed_types[0] != bool:
        allowed_types.remove(bool)
        allowed_types.insert(0, bool)

    def proto(_v: Any) -> T:
        return _record_value_to_proto(_v, allowed_types, value_proto_class)

    return {k: proto(v) for k, v in value_dict.items()}


def record_value_dict_from_proto(
    value_dict_proto: MutableMapping[str, Any]
) -> dict[str, Any]:
    """Deserialize the record value dict from ProtoBuf."""
    return {k: _record_value_from_proto(v) for k, v in value_dict_proto.items()}

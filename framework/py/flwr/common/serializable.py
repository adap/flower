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
"""Serializable base class."""


import hashlib
from functools import singledispatchmethod
from typing import TypeVar

from .constant import OBJECT_CONTENT_LEN, OBJECT_NAME_LEN, PAD_SYMBOL

T = TypeVar("T", bound="Serializable")


class Serializable:
    """Base class for serializable objects."""

    @singledispatchmethod
    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        raise NotImplementedError()

    @serialize.register
    def _(self, refs_dict: dict) -> bytes:
        """Serialize references of child objects."""
        raise NotImplementedError()

    @classmethod
    def deserialize(cls: type[T], serialized: bytes) -> T:
        """Deserialize from bytes and return an instance of the class."""
        raise NotImplementedError()


def get_object_id(serialized: bytes) -> str:
    """Return a SHA-256 hash of the serialized object."""
    return hashlib.sha256(serialized).hexdigest()


def get_object_content(serialized: bytes, cls: type[T]) -> bytes:
    """Return object content but raise an error if object type in bytes does not match
    name of class."""
    class_name = cls.__qualname__
    object_type = object_type_from_bytes(serialized)
    if not object_type == class_name:
        raise ValueError(
            f"Class name ({class_name}) and type of serialized object "
            f"({object_type}) do not match."
        )

    # Return object content by excluding the header
    return serialized[OBJECT_NAME_LEN + OBJECT_CONTENT_LEN :]


def add_header_to_object_content(object_content: bytes, cls: T) -> bytes:
    """Add header to object content."""
    # Construct header
    header = object_type_to_bytes(
        cls.__class__.__qualname__
    ) + object_content_len_to_bytes(object_content)
    # Concatenate header and object content
    return header + object_content


def _get_object_head(serialized: bytes) -> bytes:
    """Return object head from bytes."""
    return serialized[: OBJECT_NAME_LEN + OBJECT_CONTENT_LEN]


def object_type_to_bytes(class_name: str) -> bytes:
    """Return object name based on the class it is based on.

    Applies an `OBJECT_NAME_LEN` left padding with `PAD_SYMBOL`.
    """
    if len(class_name) > OBJECT_NAME_LEN:
        raise ValueError(
            f"The name of class `{class_name}` exceeds the maximum "
            f"length ({OBJECT_NAME_LEN} char)"
        )
    return class_name.encode(encoding="utf-8").ljust(OBJECT_NAME_LEN, PAD_SYMBOL)


def object_type_from_bytes(serialized: bytes) -> str:
    """Return object type from bytes."""
    obj_head = _get_object_head(serialized)
    return obj_head[:OBJECT_NAME_LEN].rstrip(PAD_SYMBOL).decode(encoding="utf-8")


def object_content_len_to_bytes(object_content: bytes) -> bytes:
    """Return size in Bytes of the bytes buffer passed."""
    return len(object_content).to_bytes(
        OBJECT_CONTENT_LEN, byteorder="little", signed=False
    )


def object_content_len_from_bytes(serialized: bytes) -> int:
    """Return length of serialized object in bytes."""
    obj_head = _get_object_head(serialized)
    return int.from_bytes(obj_head[OBJECT_NAME_LEN:], byteorder="little")

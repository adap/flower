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
from typing import TypeVar

from .constant import OBJECT_CONTENT_LEN, OBJECT_NAME_LEN, PAD_SYMBOL

T = TypeVar("T", bound="Serializable")


class Serializable:
    """Base class for serializable objects."""

    def serialize(self) -> bytes:
        """Serialize the object to bytes."""
        raise NotImplementedError()

    @classmethod
    def deserialize(cls: type[T], serialized: bytes) -> T:
        """Deserialize from bytes and return an instance of the class."""
        raise NotImplementedError()

    @property
    def object_id(self) -> str:
        """Return a SHA-256 hash of the serialized representation."""
        serialized: bytes = self.serialize()
        return hashlib.sha256(serialized).hexdigest()

    @property
    def object_name(self) -> bytes:
        """Return object name based on the class it is based on.

        Applies an `OBJECT_NAME_LEN` left padding with `PAD_SYMBOL`.
        """
        class_name = self.__class__.__qualname__.lower()
        if len(class_name) > OBJECT_NAME_LEN:
            raise ValueError(
                f"The name of class `{class_name}` exceeds the maximum "
                f"length ({OBJECT_NAME_LEN} char)"
            )
        return class_name.encode(encoding="utf-8").ljust(OBJECT_NAME_LEN, PAD_SYMBOL)


def get_object_content_size(object_content: bytes) -> bytes:
    """Return size in Bytes of the bytes buffer passed."""
    return len(object_content).to_bytes(
        OBJECT_CONTENT_LEN, byteorder="little", signed=False
    )


def _get_object_head(serialized: bytes) -> bytes:
    """Return object head from bytes."""
    return serialized[: OBJECT_NAME_LEN + OBJECT_CONTENT_LEN]


def get_object_type(serialized: bytes) -> str:
    """Return object type from bytes."""
    obj_head = _get_object_head(serialized)
    return obj_head[:OBJECT_NAME_LEN].rstrip(PAD_SYMBOL).decode(encoding="utf-8")


def get_object_content_len(serialized: bytes) -> int:
    """Return length of serialized object in bytes."""
    obj_head = _get_object_head(serialized)
    return int.from_bytes(obj_head[OBJECT_NAME_LEN:], byteorder="little")

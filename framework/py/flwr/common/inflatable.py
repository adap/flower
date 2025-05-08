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
"""Inflatable base class."""


import hashlib
from typing import TypeVar

from .constant import HEAD_BODY_DIVIDER

T = TypeVar("T", bound="Inflatable")


class Inflatable:
    """Base class for inflatable objects."""

    def deflate(self) -> tuple[bytes, str]:
        """Deflate object."""
        raise NotImplementedError()

    @property
    def object_id(self):
        """Get object_id."""
        _, object_id = self.deflate()
        return object_id

    @object_id.setter
    def object_id(self, value: str) -> None:
        self._object_id = value


def get_object_id(object_content: bytes) -> str:
    """Return a SHA-256 hash of the (deflated) object content."""
    return hashlib.sha256(object_content).hexdigest()


def get_object_body(serialized: bytes, cls: type[T]) -> bytes:
    """Return object body but raise an error if object type in bytes does not match name
    of class."""
    class_name = cls.__qualname__
    object_type = object_type_from_bytes(serialized)
    if not object_type == class_name:
        raise ValueError(
            f"Class name ({class_name}) and type of serialized object "
            f"({object_type}) do not match."
        )

    # Return object body
    return _get_object_body(serialized)


def add_header_to_object_body(object_body: bytes, cls: T) -> bytes:
    """Add header to object content."""
    # Construct header
    header = f"{cls.__class__.__qualname__} {len(object_body)}"
    enc_header = header.encode(encoding="utf-8")
    # Concatenate header and object body
    return enc_header + HEAD_BODY_DIVIDER + object_body


def _get_object_head(serialized: bytes) -> bytes:
    """Return object head from object content."""
    return serialized[: serialized.find(HEAD_BODY_DIVIDER)]


def _get_object_body(serialized: bytes) -> bytes:
    """Return object body from object content."""
    return serialized[serialized.find(HEAD_BODY_DIVIDER) + 1 :]


def object_type_from_bytes(serialized: bytes) -> str:
    """Return object type from bytes."""
    obj_head: str = _get_object_head(serialized).decode(encoding="utf-8")
    return obj_head.split(" ", 1)[0]

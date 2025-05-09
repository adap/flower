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
"""InflatableObject base class."""


import hashlib
from typing import TypeVar

from .constant import HEAD_BODY_DIVIDER, TYPE_BODY_LEN_DIVIDER

T = TypeVar("T", bound="InflatableObject")


class InflatableObject:
    """Base class for inflatable objects."""

    def deflate(self) -> bytes:
        """Deflate object."""
        raise NotImplementedError()

    @classmethod
    def inflate(cls, object_content: bytes) -> "InflatableObject":
        """Inflate the object from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content.

        Returns
        -------
        InflatableObject
            The inflated object.
        """
        raise NotImplementedError()

    @property
    def object_id(self) -> str:
        """Get object_id."""
        return get_object_id(self.deflate())


def get_object_id(object_content: bytes) -> str:
    """Return a SHA-256 hash of the (deflated) object content."""
    return hashlib.sha256(object_content).hexdigest()


def get_object_body(object_content: bytes, cls: type[T]) -> bytes:
    """Return object body but raise an error if object type doesn't match class name."""
    class_name = cls.__qualname__
    object_type = get_object_type_from_object_content(object_content)
    if not object_type == class_name:
        raise ValueError(
            f"Class name ({class_name}) and object type "
            f"({object_type}) do not match."
        )

    # Return object body
    return _get_object_body(object_content)


def add_header_to_object_body(object_body: bytes, cls: T) -> bytes:
    """Add header to object content."""
    # Construct header
    header = f"{cls.__class__.__qualname__}{TYPE_BODY_LEN_DIVIDER}{len(object_body)}"
    enc_header = header.encode(encoding="utf-8")
    # Concatenate header and object body
    return enc_header + HEAD_BODY_DIVIDER + object_body


def _get_object_head(object_content: bytes) -> bytes:
    """Return object head from object content."""
    return object_content.split(HEAD_BODY_DIVIDER, 1)[0]


def _get_object_body(object_content: bytes) -> bytes:
    """Return object body from object content."""
    return object_content.split(HEAD_BODY_DIVIDER, 1)[1]


def get_object_type_from_object_content(object_content: bytes) -> str:
    """Return object type from bytes."""
    obj_head: str = _get_object_head(object_content).decode(encoding="utf-8")
    return obj_head.split(TYPE_BODY_LEN_DIVIDER, 1)[0]

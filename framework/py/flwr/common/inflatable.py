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


from __future__ import annotations

import hashlib
from logging import ERROR
from typing import TypeVar, cast

from .constant import HEAD_BODY_DIVIDER, HEAD_VALUE_DIVIDER
from .logger import log


class UnexpectedObjectContentError(Exception):
    """Exception raised when the content of an object does not conform to the expected
    structure for an InflatableObject (i.e., head, body, and values within the head)."""

    def __init__(self, object_id: str, reason: str):
        super().__init__(
            f"Object with ID '{object_id}' has an unexpected structure. {reason}"
        )


class InflatableObject:
    """Base class for inflatable objects."""

    def deflate(self) -> bytes:
        """Deflate object."""
        raise NotImplementedError()

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> InflatableObject:
        """Inflate the object from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content.

        children : Optional[dict[str, InflatableObject]] (default: None)
            Dictionary of children InflatableObjects mapped to their object IDs. These
            childrens enable the full inflation of the parent InflatableObject.

        Returns
        -------
        InflatableObject
            The inflated object.
        """
        raise NotImplementedError()

    @property
    def object_id(self) -> str:
        """Get object_id."""
        if self.is_dirty or "_object_id" not in self.__dict__:
            self.__dict__["_object_id"] = get_object_id(self.deflate())
        return cast(str, self.__dict__["_object_id"])

    @property
    def children(self) -> dict[str, InflatableObject] | None:
        """Get all child objects as a dictionary or None if there are no children."""
        return None

    @property
    def is_dirty(self) -> bool:
        """Check if the object is dirty after the last deflation.

        An object is considered dirty if its content has changed since the last its
        object ID was computed.
        """
        return True


T = TypeVar("T", bound=InflatableObject)


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


def add_header_to_object_body(object_body: bytes, obj: InflatableObject) -> bytes:
    """Add header to object content."""
    # Construct header
    header = f"%s{HEAD_VALUE_DIVIDER}%s{HEAD_VALUE_DIVIDER}%d" % (
        obj.__class__.__qualname__,  # Type of object
        ",".join((obj.children or {}).keys()),  # IDs of child objects
        len(object_body),  # Length of object body
    )

    # Concatenate header and object body
    ret = bytearray()
    ret.extend(header.encode(encoding="utf-8"))
    ret.extend(HEAD_BODY_DIVIDER)
    ret.extend(object_body)
    return bytes(ret)


def _get_object_head(object_content: bytes) -> bytes:
    """Return object head from object content."""
    index = object_content.find(HEAD_BODY_DIVIDER)
    return object_content[:index]


def _get_object_body(object_content: bytes) -> bytes:
    """Return object body from object content."""
    index = object_content.find(HEAD_BODY_DIVIDER)
    return object_content[index + len(HEAD_BODY_DIVIDER) :]


def is_valid_sha256_hash(object_id: str) -> bool:
    """Check if the given string is a valid SHA-256 hash.

    Parameters
    ----------
    object_id : str
        The string to check.

    Returns
    -------
    bool
        ``True`` if the string is a valid SHA-256 hash, ``False`` otherwise.
    """
    if len(object_id) != 64:
        return False
    try:
        # If base 16 int conversion succeeds, it's a valid hexadecimal str
        int(object_id, 16)
        return True
    except ValueError:
        return False


def get_object_type_from_object_content(object_content: bytes) -> str:
    """Return object type from bytes."""
    return get_object_head_values_from_object_content(object_content)[0]


def get_object_children_ids_from_object_content(object_content: bytes) -> list[str]:
    """Return object children IDs from bytes."""
    return get_object_head_values_from_object_content(object_content)[1]


def get_object_body_len_from_object_content(object_content: bytes) -> int:
    """Return length of the object body."""
    return get_object_head_values_from_object_content(object_content)[2]


def check_body_len_consistency(object_content: bytes) -> bool:
    """Check that the object body is of length as specified in the head."""
    try:
        body_len = get_object_body_len_from_object_content(object_content)
        return body_len == len(_get_object_body(object_content))
    except ValueError:
        log(ERROR, "Object content does match the expected format.")
        return False


def get_object_head_values_from_object_content(
    object_content: bytes,
) -> tuple[str, list[str], int]:
    """Return object type and body length from object content.

    Parameters
    ----------
    object_content : bytes
        The deflated object content.

    Returns
    -------
    tuple[str, list[str], int]
        A tuple containing:
        - The object type as a string.
        - A list of child object IDs as strings.
        - The length of the object body as an integer.
    """
    head = _get_object_head(object_content).decode(encoding="utf-8")
    obj_type, children_str, body_len = head.split(HEAD_VALUE_DIVIDER)
    children_ids = children_str.split(",") if children_str else []
    return obj_type, children_ids, int(body_len)


def get_descendant_object_ids(obj: InflatableObject) -> set[str]:
    """Get a set of object IDs of all descendants."""
    descendants = set(get_all_nested_objects(obj).keys())
    # Exclude Object ID of parent object
    descendants.discard(obj.object_id)
    return descendants


def get_all_nested_objects(obj: InflatableObject) -> dict[str, InflatableObject]:
    """Get a dictionary of all nested objects, including the object itself.

    Each key in the dictionary is an object ID, and the entries are ordered by post-
    order traversal, i.e., child objects appear before their respective parents.
    """
    ret: dict[str, InflatableObject] = {}
    if children := obj.children:
        for child in children.values():
            ret.update(get_all_nested_objects(child))

    ret[obj.object_id] = obj

    return ret

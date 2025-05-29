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
"""Inflatable test."""


from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from .constant import HEAD_BODY_DIVIDER, HEAD_VALUE_DIVIDER
from .inflatable import (
    InflatableObject,
    _get_object_body,
    _get_object_head,
    add_header_to_object_body,
    check_body_len_consistency,
    get_desdendant_object_ids,
    get_object_body,
    get_object_body_len_from_object_content,
    get_object_head_values_from_object_content,
    get_object_id,
    get_object_type_from_object_content,
    is_valid_sha256_hash,
)


@dataclass
class CustomDataClass(InflatableObject):
    """A dummy dataclass to test Inflatable features."""

    data: bytes
    _children = None

    def deflate(self) -> bytes:  # noqa: D102
        obj_body = self.data
        return add_header_to_object_body(object_body=obj_body, obj=self)

    @classmethod
    def inflate(  # noqa: D102
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> CustomDataClass:
        if children is not None:
            raise ValueError("`CustomDataClass` does not have children.")
        object_body = get_object_body(object_content, cls)
        return cls(data=object_body)

    @property
    def children(self) -> dict[str, InflatableObject] | None:
        """Get children."""
        return self._children

    @children.setter
    def children(self, children: dict[str, InflatableObject]) -> None:
        """Set children only for testing purposes."""
        self._children = children


def test_deflate_and_inflate() -> None:
    """Deflate a custom object and verify its ``object_id``.

    Then inflate it and verify the content is identical to the original object.
    """
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    # assert
    # Class name matches
    assert get_object_type_from_object_content(obj_b) == obj.__class__.__qualname__
    # Content length matches
    assert len(get_object_body(obj_b, CustomDataClass)) == len(data)

    # assert
    # both objects are identical
    assert get_object_id(obj_b) == obj.object_id

    # Inflate and check object payload is the same
    obj_ = CustomDataClass.inflate(obj_b)
    assert obj_.data == obj.data

    # Assert
    # Inflate passing children raises ValueError
    with pytest.raises(ValueError):
        CustomDataClass.inflate(obj_b, children={"1234": obj})


def test_get_object_id() -> None:
    """Test helper function to get object id from bytes."""
    some_bytes = b"hello world"
    expected = hashlib.sha256(some_bytes).hexdigest()
    assert get_object_id(some_bytes) == expected


def test_get_object_body() -> None:
    """Test helper function to extract object body from object content."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    assert _get_object_body(obj_b) == data
    assert get_object_body(obj_b, CustomDataClass) == data


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_add_header_to_object_body(children: list[InflatableObject]) -> None:
    """Test helper function that adds the header to the object body."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj.children = {child.object_id: child for child in children}
    obj_b = obj.deflate()

    # Expected object head
    head_str = f"%s{HEAD_VALUE_DIVIDER}%s{HEAD_VALUE_DIVIDER}%d" % (
        "CustomDataClass",
        ",".join(child.object_id for child in children),
        len(data),
    )
    exp_obj_head = head_str.encode()
    assert _get_object_head(obj_b) == exp_obj_head

    # Expected object content
    exp_obj_content = exp_obj_head + HEAD_BODY_DIVIDER + data
    assert add_header_to_object_body(data, obj) == exp_obj_content


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_get_head_values_from_object_content(children: list[InflatableObject]) -> None:
    """Test helper function that extracts the values of the object head."""
    # Prepare
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj.children = {child.object_id: child for child in children}
    obj_b = obj.deflate()

    # Execute
    obj_type, children_ids, body_len = get_object_head_values_from_object_content(obj_b)

    # Assert
    assert obj_type == "CustomDataClass"
    assert children_ids == [child.object_id for child in children]
    assert body_len == len(data)


def test_get_object_type_from_object_content() -> None:
    """Test helper function that extracts the name of the class of the deflated
    object."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    assert get_object_type_from_object_content(obj_b) == obj.__class__.__qualname__


def test_is_valid_sha256_hash_valid() -> None:
    """Test helper function that checks if a string is a valid SHA256 hash."""
    # Prepare
    valid_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # Execute & assert
    assert is_valid_sha256_hash(valid_hash)


def test_is_valid_sha256_hash_invalid() -> None:
    """Test helper function that checks if a string is a valid SHA256 hash."""
    # Prepare
    invalid_hash = "invalid_hash"

    # Execute & assert
    assert not is_valid_sha256_hash(invalid_hash)


def test_check_body_length() -> None:
    """Test helper function that checks if the specified body length in the object head
    matches the actual length of the object body."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    # Body length is measured correctly
    assert get_object_body_len_from_object_content(obj_b) == len(data)

    # Consistent: passes
    assert check_body_len_consistency(obj_b)

    # Extend content artificially
    obj_b_ = obj_b + b"more content"
    # Inconsistent: fails
    assert not check_body_len_consistency(obj_b_)


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_get_desdendants(children: list[InflatableObject]) -> None:
    """Test computing list of object IDs for all descendants."""
    data = b"this is a test"
    obj = CustomDataClass(data)

    obj.children = {child.object_id: child for child in children}

    # Assert
    assert get_desdendant_object_ids(obj) == {child.object_id for child in children}

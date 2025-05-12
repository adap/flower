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

import hashlib
from dataclasses import dataclass

import pytest

from .constant import HEAD_BODY_DIVIDER, TYPE_BODY_LEN_DIVIDER
from .inflatable import (
    InflatableObject,
    _get_object_body,
    _get_object_head,
    add_header_to_object_body,
    get_object_body,
    get_object_id,
    get_object_type_from_object_content,
)


@dataclass
class CustomDataClass(InflatableObject):
    """A dummy dataclass to test Inflatable features."""

    data: bytes

    def deflate(self) -> bytes:  # noqa: D102
        obj_body = self.data
        return add_header_to_object_body(object_body=obj_body, cls=self)

    @classmethod
    def inflate(  # noqa: D102
        cls, object_content: bytes, children: dict[str, InflatableObject]
    ) -> "CustomDataClass":

        if children:
            raise ValueError("`CustomDataClass` does not have children.")
        object_body = get_object_body(object_content, cls)
        return cls(data=object_body)


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
    obj_ = CustomDataClass.inflate(obj_b, {})
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


def test_add_header_to_object_body() -> None:
    """Test helper function that adds the header to the object body and returns the
    object content."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    # Expected object head
    exp_obj_head = f"CustomDataClass{TYPE_BODY_LEN_DIVIDER}{len(data)}".encode()
    assert _get_object_head(obj_b) == exp_obj_head

    # Expected object content
    exp_obj_conten = exp_obj_head + HEAD_BODY_DIVIDER + data
    assert add_header_to_object_body(data, obj) == exp_obj_conten


def test_get_object_type_from_object_content() -> None:
    """Test helper function that extracts the name of the class of the deflated
    object."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    assert get_object_type_from_object_content(obj_b) == obj.__class__.__qualname__

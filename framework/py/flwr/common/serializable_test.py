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
"""Serializable test."""


from dataclasses import dataclass

from .serializable import (
    Serializable,
    add_header_to_object_body,
    get_object_body,
    get_object_id,
    object_type_from_bytes,
)


@dataclass
class CustomDataClass(Serializable):
    """A dummy dataclass to test Serializable features."""

    data: bytes

    def serialize(self) -> bytes:  # noqa: D102
        obj_body = self.data
        return add_header_to_object_body(object_body=obj_body, cls=self)

    @classmethod
    def deserialize(cls, serialized: bytes) -> "CustomDataClass":  # noqa: D102
        data = get_object_body(serialized, cls)
        return CustomDataClass(data)


def test_serialization_and_deserialization() -> None:
    """Serialize a custom object and deserialize it."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.serialize()

    obj.serialize()

    # assert
    # Class name matches
    assert object_type_from_bytes(obj_b) == obj.__class__.__qualname__
    # Content length matches
    assert len(get_object_body(obj_b, CustomDataClass)) == len(data)

    # Deserialize
    obj_ = CustomDataClass.deserialize(obj_b)

    # assert
    # both objects are identical
    assert get_object_id(obj_b) == get_object_id(obj_.serialize())

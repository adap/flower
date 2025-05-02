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
    add_header_to_object_content,
    get_object_content,
    object_content_len_from_bytes,
    object_type_from_bytes,
)


@dataclass
class CustomDataClass(Serializable):
    """A dummy dataclass to test Serializable features."""

    data: bytes

    def serialize(self) -> bytes:  # noqa: D102
        obj_content = self.data
        return add_header_to_object_content(
            object_content=obj_content, class_name=self.__class__.__qualname__
        )

    @classmethod
    def deserialize(cls, serialized: bytes) -> "CustomDataClass":  # noqa: D102
        data = get_object_content(serialized, class_name=cls.__qualname__)
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
    assert object_content_len_from_bytes(obj_b) == len(data)

    # Deserialize
    obj_ = CustomDataClass.deserialize(obj_b)

    # assert
    # both objects are identical
    assert obj.object_id == obj_.object_id

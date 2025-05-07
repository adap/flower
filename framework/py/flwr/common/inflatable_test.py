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


from dataclasses import dataclass

from .inflatable import (
    Inflatable,
    add_header_to_object_body,
    get_object_body,
    get_object_id,
    object_type_from_bytes,
)


@dataclass
class CustomDataClass(Inflatable):
    """A dummy dataclass to test Inflatable features."""

    data: bytes

    def deflate(self) -> tuple[bytes, str]:  # noqa: D102
        obj_body = self.data
        obj_content = add_header_to_object_body(object_body=obj_body, cls=self)
        return obj_content, get_object_id(obj_content)


def test_deflate() -> None:
    """Deflate a custom object and verify its object_id it."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b, _ = obj.deflate()

    # assert
    # Class name matches
    assert object_type_from_bytes(obj_b) == obj.__class__.__qualname__
    # Content length matches
    assert len(get_object_body(obj_b, CustomDataClass)) == len(data)

    # assert
    # both objects are identical
    assert get_object_id(obj_b) == obj.object_id

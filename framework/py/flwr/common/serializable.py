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
"""Serializable ABC."""


import hashlib
from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """ABC class for serializable objects."""

    pad_symbol = b"*"
    obj_name_len = 16
    obj_content_len = 8

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the object to bytes."""

    @classmethod
    @abstractmethod
    def deserialize(cls: type[T], serialized: bytes) -> T:
        """Deserialize from bytes and return an instance of the class."""

    @property
    def object_id(self) -> str:
        """Return a SHA-256 hash of the serialized representation."""
        serialized: bytes = self.serialize()
        return hashlib.sha256(serialized).hexdigest()

    @property
    def object_name(self) -> bytes:
        """Return object name based on the class it is based on.

        Applies an `obj_name_len` left padding with symbol '*'.
        """
        class_name = self.__class__.__name__.lower()
        return class_name.encode().ljust(self.obj_name_len, self.pad_symbol)

    def get_object_content_size(self, object_content: bytes) -> bytes:
        """Return size in Bytes of the bytes buffer passed."""
        return len(object_content).to_bytes(
            self.obj_content_len, byteorder="little", signed=False
        )

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

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize the object to bytes."""

    @classmethod
    @abstractmethod
    def deserialize(cls: type[T], serialized: bytes) -> T:
        """Deserialize from bytes and return an instance of the class."""

    @staticmethod
    def concatenate(bytes_list: list[bytes]) -> bytes:
        """Add Bytes divider between each Bytes segement and concatenate them."""
        divider = b"\x00"
        return divider.join(bytes_list)

    @property
    def object_id(self) -> str:
        """Return a SHA-256 hash of the serialized representation."""
        serialized: bytes = self.serialize()
        return hashlib.sha256(serialized).hexdigest()

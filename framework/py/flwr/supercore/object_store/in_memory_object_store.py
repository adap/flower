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
"""Flower in-memory ObjectStore implementation."""


from typing import Optional

from flwr.common.inflatable import get_object_id, is_valid_sha256_hash

from .object_store import ObjectStore


class InMemoryObjectStore(ObjectStore):
    """In-memory implementation of the ObjectStore interface."""

    def __init__(self, verify: bool = True) -> None:
        self.verify = verify
        self.store: dict[str, bytes] = {}

    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store."""
        # Verify object ID format (must be a valid sha256 hash)
        if not is_valid_sha256_hash(object_id):
            raise ValueError(f"Invalid object ID format: {object_id}")

        # Verify object_id and object_content match
        if self.verify:
            object_id_from_content = get_object_id(object_content)
            if object_id != object_id_from_content:
                raise ValueError(f"Object ID {object_id} does not match content hash")

        # Return if object is already present in the store
        if object_id in self.store:
            return

        self.store[object_id] = object_content

    def get(self, object_id: str) -> Optional[bytes]:
        """Get an object from the store."""
        return self.store.get(object_id)

    def delete(self, object_id: str) -> None:
        """Delete an object from the store."""
        if object_id in self.store:
            del self.store[object_id]

    def clear(self) -> None:
        """Clear the store."""
        self.store.clear()

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        return object_id in self.store

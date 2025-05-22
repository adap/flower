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
"""Tests for factory class that creates ObjectStore instances."""


import unittest

from .in_memory_object_store import InMemoryObjectStore
from .object_store_factory import ObjectStoreFactory


class TestObjectStoreFactory(unittest.TestCase):
    """Test the ObjectStoreFactory class."""

    def test_store_creates_in_memory_object_store(self) -> None:
        """Test that the factory creates an InMemoryObjectStore instance."""
        factory = ObjectStoreFactory()
        store = factory.store()
        self.assertIsInstance(store, InMemoryObjectStore)

    def test_store_returns_same_instance(self) -> None:
        """Test that the factory returns the same instance on subsequent calls."""
        factory = ObjectStoreFactory()
        store1 = factory.store()
        store2 = factory.store()
        self.assertIs(store1, store2)


if __name__ == "__main__":
    unittest.main()

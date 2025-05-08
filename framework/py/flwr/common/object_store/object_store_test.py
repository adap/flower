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
"""Tests for ObjectStore."""


import unittest
from abc import abstractmethod

from .object_store import ObjectStore


class ObjectStoreTest(unittest.TestCase):
    """Test all ObjectStore implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def object_store_factory(self) -> ObjectStore:
        """Provide ObjectStore implementation to test."""
        raise NotImplementedError()

    def test_get_non_existent_key(self) -> None:
        """Test get method with a non-existent key."""
        object_store = self.object_store_factory()
        key = "non_existent_key"

        retrieved_value = object_store.get(key)

        self.assertIsNone(retrieved_value)

    def test_put_and_get(self) -> None:
        """Test put and get methods."""
        object_store = self.object_store_factory()
        key = "test_key"
        object_content = b"test_value"

        object_store.put(key, object_content)
        retrieved_value = object_store.get(key)

        self.assertEqual(object_content, retrieved_value)

    def test_put_overwrite(self) -> None:
        """Test put method with an existing key."""
        object_store = self.object_store_factory()
        key = "test_key"
        object_content1 = b"test_value1"
        object_content2 = b"test_value2"

        object_store.put(key, object_content1)
        object_store.put(key, object_content2)
        retrieved_value = object_store.get(key)

        self.assertEqual(object_content2, retrieved_value)

    def test_put_empty_key(self) -> None:
        """Test put method with an empty key."""
        object_store = self.object_store_factory()
        key = ""
        object_content = b"test_value"

        object_store.put(key, object_content)
        retrieved_value = object_store.get(key)

        self.assertEqual(object_content, retrieved_value)

    def test_delete(self) -> None:
        """Test delete method."""
        object_store = self.object_store_factory()
        key = "test_key"
        object_content = b"test_value"

        object_store.put(key, object_content)
        object_store.delete(key)
        retrieved_value = object_store.get(key)

        self.assertIsNone(retrieved_value)

    def test_delete_non_existent_key(self) -> None:
        """Test delete method with a non-existent key."""
        object_store = self.object_store_factory()
        key = "non_existent_key"

        object_store.delete(key)
        # No exception should be raised

    def test_clear(self) -> None:
        """Test clear method."""
        object_store = self.object_store_factory()
        key1 = "test_key1"
        object_content1 = b"test_value1"
        key2 = "test_key2"
        object_content2 = b"test_value2"

        object_store.put(key1, object_content1)
        object_store.put(key2, object_content2)
        object_store.clear()

        retrieved_value1 = object_store.get(key1)
        retrieved_value2 = object_store.get(key2)

        self.assertIsNone(retrieved_value1)
        self.assertIsNone(retrieved_value2)

    def test_clear_empty_store(self) -> None:
        """Test clear method on an empty store."""
        object_store = self.object_store_factory()

        object_store.clear()
        # No exception should be raised

    def test_contains(self) -> None:
        """Test __contains__ method."""
        object_store = self.object_store_factory()
        key = "test_key"
        object_content = b"test_value"

        object_store.put(key, object_content)

        self.assertTrue(key in object_store)
        self.assertFalse("non_existent_key" in object_store)

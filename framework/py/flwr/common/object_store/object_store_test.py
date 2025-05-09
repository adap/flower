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

    def test_get_non_existent_object_id(self) -> None:
        """Test get method with a non-existent object_id."""
        object_store = self.object_store_factory()
        object_id = "non_existent_object_id"

        retrieved_value = object_store.get(object_id)

        self.assertIsNone(retrieved_value)

    def test_put_and_get(self) -> None:
        """Test put and get methods."""
        object_store = self.object_store_factory()
        object_id = "test_object_id"
        object_content = b"test_value"

        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        self.assertEqual(object_content, retrieved_value)

    def test_put_overwrite(self) -> None:
        """Test put method with an existing object_id."""
        object_store = self.object_store_factory()
        object_id = "test_object_id"
        object_content1 = b"test_value1"
        object_content2 = b"test_value2"

        object_store.put(object_id, object_content1)
        object_store.put(object_id, object_content2)
        retrieved_value = object_store.get(object_id)

        self.assertEqual(object_content2, retrieved_value)

    def test_put_empty_object_id(self) -> None:
        """Test put method with an empty object_id."""
        object_store = self.object_store_factory()
        object_id = ""
        object_content = b"test_value"

        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        self.assertEqual(object_content, retrieved_value)

    def test_delete(self) -> None:
        """Test delete method."""
        object_store = self.object_store_factory()
        object_id = "test_object_id"
        object_content = b"test_value"

        object_store.put(object_id, object_content)
        object_store.delete(object_id)
        retrieved_value = object_store.get(object_id)

        self.assertIsNone(retrieved_value)

    def test_delete_non_existent_object_id(self) -> None:
        """Test delete method with a non-existent object_id."""
        object_store = self.object_store_factory()
        object_id = "non_existent_object_id"

        object_store.delete(object_id)
        # No exception should be raised

    def test_clear(self) -> None:
        """Test clear method."""
        object_store = self.object_store_factory()
        object_id1 = "test_object_id1"
        object_content1 = b"test_value1"
        object_id2 = "test_object_id2"
        object_content2 = b"test_value2"

        object_store.put(object_id1, object_content1)
        object_store.put(object_id2, object_content2)
        object_store.clear()

        retrieved_value1 = object_store.get(object_id1)
        retrieved_value2 = object_store.get(object_id2)

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
        object_id = "test_object_id"
        object_content = b"test_value"

        object_store.put(object_id, object_content)

        self.assertTrue(object_id in object_store)
        self.assertFalse("non_existent_object_id" in object_store)

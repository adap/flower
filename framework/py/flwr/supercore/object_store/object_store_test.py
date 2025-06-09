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

from parameterized import parameterized

from flwr.common.inflatable import get_object_id
from flwr.common.inflatable_test import CustomDataClass

from .in_memory_object_store import InMemoryObjectStore
from .object_store import NoObjectInStoreError, ObjectStore


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
        # Prepare
        object_store = self.object_store_factory()
        object_id = "non_existent_object_id"

        # Execute
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_put_and_get(self) -> None:
        """Test put and get methods."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(object_ids=[object_id])

        # Execute
        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertEqual(object_content, retrieved_value)

    def test_put_overwrite(self) -> None:
        """Test put method with an existing object_id."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(object_ids=[object_id])

        # Execute
        object_store.put(object_id, object_content)
        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertEqual(object_content, retrieved_value)

    def test_put_object_id_and_content_pair_not_matching(self) -> None:
        """Test put method with an object_id that does not match that of content."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        object_store.preregister(object_ids=[object_id])

        # Execute
        try:
            object_store.put(object_id, object_content)
            # Assert
            raise AssertionError("Expected ValueError not raised")
        except ValueError:
            # Assert
            assert True

    def test_delete(self) -> None:
        """Test delete method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(object_ids=[object_id])
        object_store.put(object_id, object_content)

        # Execute
        object_store.delete(object_id)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_delete_non_existent_object_id(self) -> None:
        """Test delete method with a non-existent object_id."""
        # Prepare
        object_store = self.object_store_factory()
        object_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        object_store.delete(object_id)
        # No exception should be raised

    def test_clear(self) -> None:
        """Test clear method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value1")
        object_content1 = obj.deflate()
        object_id1 = get_object_id(object_content1)
        object_store.preregister(object_ids=[object_id1])
        obj = CustomDataClass(data=b"test_value2")
        object_content2 = obj.deflate()
        object_id2 = get_object_id(object_content2)
        object_store.preregister(object_ids=[object_id2])

        object_store.put(object_id1, object_content1)
        object_store.put(object_id2, object_content2)

        # Execute
        object_store.clear()

        # Assert
        retrieved_value1 = object_store.get(object_id1)
        retrieved_value2 = object_store.get(object_id2)

        self.assertIsNone(retrieved_value1)
        self.assertIsNone(retrieved_value2)

    def test_clear_empty_store(self) -> None:
        """Test clear method on an empty store."""
        # Prepare
        object_store = self.object_store_factory()

        # Execute
        object_store.clear()
        # No exception should be raised

    def test_contains(self) -> None:
        """Test __contains__ method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value1")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(object_ids=[object_id])
        object_store.put(object_id, object_content)
        unavailable = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        # Execute
        contained = object_id in object_store
        not_contained = unavailable in object_store

        # Assert
        self.assertTrue(contained)
        self.assertFalse(not_contained)

    def test_put_without_preregistering(self) -> None:
        """Test put without preregistering first."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)

        # Execute
        with self.assertRaises(NoObjectInStoreError):
            object_store.put(object_id, object_content)

    def test_preregister(self) -> None:
        """Test preregister functionality."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value1")
        object_content1 = obj.deflate()
        object_id1 = get_object_id(object_content1)
        obj = CustomDataClass(data=b"test_value2")
        object_content2 = obj.deflate()
        object_id2 = get_object_id(object_content2)

        # Execute (preregister all)
        not_present = object_store.preregister(object_ids=[object_id1, object_id2])

        # Assert (none was present)
        self.assertEqual([object_id1, object_id2], not_present)

        obj = CustomDataClass(data=b"test_value3")
        object_content3 = obj.deflate()
        object_id3 = get_object_id(object_content3)
        # Execute (preregister new object)
        not_present = object_store.preregister(object_ids=[object_id3])
        # Assert (only new message is not present)
        self.assertEqual([object_id3], not_present)

    @parameterized.expand([(""), ("invalid")])  # type: ignore
    def test_preregister_with_invalid_object_id(self, invalid_object_id) -> None:
        """Test preregistering with object_id that is not a valid SHA256."""
        # Prepare
        object_store = self.object_store_factory()

        # Execute
        with self.assertRaises(ValueError):
            object_store.preregister(object_ids=[invalid_object_id])

    def test_get_message_descendants_ids(self) -> None:
        """Test setting and getting mapping of message object id and its descendants."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)

        # Execute
        # Insert
        object_store.set_message_descendant_ids(
            msg_object_id=object_id, descendant_ids=[]
        )
        # Extract correct
        descendant_ids = object_store.get_message_descendant_ids(
            msg_object_id=object_id
        )

        # Assert
        assert descendant_ids == []

        # Extract nonexistent id
        with self.assertRaises(NoObjectInStoreError):
            object_store.get_message_descendant_ids(msg_object_id="1234")


class InMemoryStateTest(ObjectStoreTest):
    """Test InMemoryObjectStore implementation."""

    __test__ = True

    def object_store_factory(self) -> ObjectStore:
        """Return InMemoryObjectStore."""
        return InMemoryObjectStore()

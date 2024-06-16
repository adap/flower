# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Tests all Ffs implemenations have to conform to."""
# pylint: disable=invalid-name, disable=R0904

import hashlib
import json
import os
import tempfile
import unittest
from abc import abstractmethod

from flwr.server.superlink.ffs import DiskFfs, Ffs


class FfsTest(unittest.TestCase):
    """Test all ffs implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def ffs_factory(self) -> Ffs:
        """Provide Ffs implementation to test."""
        raise NotImplementedError()

    def test_put(self) -> None:
        """Test if object can be stored."""
        # Prepare
        ffs: Ffs = self.ffs_factory()
        content = b"content"
        hash_expected = hashlib.sha256(content).hexdigest()

        # Execute
        hash_actual = ffs.put(b"content", {"meta": "data"})

        # Assert
        assert isinstance(hash_actual, str)
        assert len(hash_actual) == 64
        assert hash_actual == hash_expected

        # Check if file was created
        assert [hash_expected, f"{hash_expected}.META"] == os.listdir(self.tmp_dir.name)

    def test_get(self) -> None:
        """Test if object can be retrieved."""
        # Prepare
        ffs: Ffs = self.ffs_factory()
        content_expected = b"content"
        hash_expected = hashlib.sha256(content_expected).hexdigest()
        meta_expected = {}

        with open(os.path.join(self.tmp_dir.name, hash_expected), "wb") as file:
            file.write(content_expected)

        with open(
            os.path.join(self.tmp_dir.name, f"{hash_expected}.META"), "w"
        ) as file:
            json.dump(meta_expected, file)

        # Execute
        content_actual, meta_actual = ffs.get(hash_expected)

        # Assert
        assert content_actual == content_expected
        assert meta_actual == meta_expected

    def test_delete(self) -> None:
        """Test if object can be deleted."""
        # Prepare
        ffs: Ffs = self.ffs_factory()
        content_expected = b"content"
        hash_expected = hashlib.sha256(content_expected).hexdigest()
        meta_expected = {}

        with open(os.path.join(self.tmp_dir.name, hash_expected), "wb") as file:
            file.write(content_expected)

        with open(
            os.path.join(self.tmp_dir.name, f"{hash_expected}.META"), "w"
        ) as file:
            json.dump(meta_expected, file)

        # Execute
        ffs.delete(hash_expected)

        # Assert
        assert [] == os.listdir(self.tmp_dir.name)

    def test_list(self) -> None:
        """Test if object hashes can be listed."""
        # Prepare
        ffs: Ffs = self.ffs_factory()
        content_expected = b"content"
        hash_expected = hashlib.sha256(content_expected).hexdigest()
        meta_expected = {}

        with open(os.path.join(self.tmp_dir.name, hash_expected), "wb") as file:
            file.write(content_expected)

        with open(
            os.path.join(self.tmp_dir.name, f"{hash_expected}.META"), "w"
        ) as file:
            json.dump(meta_expected, file)

        # Execute
        hashes = ffs.list()

        # Assert
        assert [hash_expected] == hashes


class DiskFfsTest(FfsTest, unittest.TestCase):
    """Test DiskFfs implementation."""

    __test__ = True

    def ffs_factory(self) -> DiskFfs:
        """Return SqliteState with file-based database."""
        # pylint: disable-next=consider-using-with,attribute-defined-outside-init
        self.tmp_dir = tempfile.TemporaryDirectory()
        ffs = DiskFfs(self.tmp_dir.name)
        return ffs


if __name__ == "__main__":
    unittest.main(verbosity=2)

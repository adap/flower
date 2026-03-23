# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for ObjectStore-backed FFS."""


import os
import sqlite3
import tempfile

import pytest

from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME

from .objectstore_ffs import ObjectStoreFfs
from .objectstore_ffs_factory import ObjectStoreFfsFactory


def test_put_get_roundtrip_in_memory() -> None:
    """Test in-memory put/get roundtrip with metadata."""
    ffs = ObjectStoreFfs(FLWR_IN_MEMORY_DB_NAME)

    content = b"fab-content"
    meta = {"signature": "ok", "issuer": "test"}

    key = ffs.put(content, meta)
    result = ffs.get(key)

    assert result == (content, meta)


def test_put_get_roundtrip_sql() -> None:
    """Test SQL put/get roundtrip with metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "state.db")
        ffs = ObjectStoreFfs(db_path)

        content = b"fab-content"
        meta = {"signature": "ok", "issuer": "test"}

        key = ffs.put(content, meta)
        result = ffs.get(key)

        assert result == (content, meta)


def test_put_is_idempotent_for_same_content_and_metadata() -> None:
    """Test idempotent behavior for repeated writes of same FAB."""
    ffs = ObjectStoreFfs(FLWR_IN_MEMORY_DB_NAME)

    content = b"same-fab"
    meta = {"k": "v"}

    key_1 = ffs.put(content, meta)
    key_2 = ffs.put(content, meta)

    assert key_1 == key_2
    assert ffs.get(key_1) == (content, meta)


def test_put_rejects_conflicting_metadata_for_same_content_hash() -> None:
    """Test hash-key conflict protection when metadata differs."""
    ffs = ObjectStoreFfs(FLWR_IN_MEMORY_DB_NAME)

    content = b"same-fab"
    ffs.put(content, {"k": "v1"})

    with pytest.raises(ValueError):
        _ = ffs.put(content, {"k": "v2"})


def test_get_missing_hash_returns_none() -> None:
    """Test missing hash behavior."""
    ffs = ObjectStoreFfs(FLWR_IN_MEMORY_DB_NAME)

    assert ffs.get("f" * 64) is None


def test_get_returns_none_for_corrupted_sql_verifications() -> None:
    """Test corrupt verification payload handling on SQL backend."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "state.db")
        ffs = ObjectStoreFfs(db_path)

        content = b"fab-content"
        key = ffs.put(content, {"issuer": "ok"})

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE fab_objects SET verifications_json = ? WHERE fab_hash = ?",
                ("not-json", key),
            )
            conn.commit()

        assert ffs.get(key) is None


def test_get_returns_none_for_corrupted_sql_content_hash() -> None:
    """Test corrupt content handling on SQL backend."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "state.db")
        ffs = ObjectStoreFfs(db_path)

        key = ffs.put(b"fab-content", {"issuer": "ok"})

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE fab_objects SET content = ? WHERE fab_hash = ?",
                (b"different-content", key),
            )
            conn.commit()

        assert ffs.get(key) is None


def test_cross_replica_sql_read_after_write() -> None:
    """Test FAB retrieval across independent factory instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "state.db")
        content = b"fab-shared"
        meta = {"issuer": "shared"}

        ffs_a = ObjectStoreFfsFactory(db_path).ffs()
        ffs_b = ObjectStoreFfsFactory(db_path).ffs()

        key = ffs_a.put(content, meta)

        assert ffs_b.get(key) == (content, meta)

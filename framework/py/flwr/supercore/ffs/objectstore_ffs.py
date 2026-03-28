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
"""ObjectStore-backed Flower File Storage for FAB content."""


import hashlib
import json
import threading
from logging import DEBUG, ERROR

from sqlalchemy import MetaData

from flwr.common.logger import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

from .ffs import Ffs


def _to_verifications_json(meta: dict[str, str]) -> str:
    """Serialize verification metadata to a canonical JSON string."""
    return json.dumps(meta, sort_keys=True, separators=(",", ":"))


def _from_verifications_json(raw: str) -> dict[str, str] | None:
    """Deserialize and validate verification metadata JSON."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(value, dict):
        return None

    meta: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not isinstance(val, str):
            return None
        meta[key] = val

    return meta


class _InMemoryObjectStoreFfs(Ffs):
    """In-memory FAB storage implementation."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.RLock()

    def put(self, content: bytes, meta: dict[str, str]) -> str:
        """Store bytes and metadata and return key (hash of content)."""
        content_hash = hashlib.sha256(content).hexdigest()
        verifications_json = _to_verifications_json(meta)

        with self._lock:
            if content_hash not in self._store:
                self._store[content_hash] = (content, verifications_json)
                return content_hash

            existing_content, existing_json = self._store[content_hash]
            existing_meta = _from_verifications_json(existing_json)

            if existing_content != content or existing_meta != meta:
                raise ValueError(
                    "Conflicting FAB content or verification metadata detected "
                    f"for hash '{content_hash}'."
                )

        return content_hash

    def get(self, key: str) -> tuple[bytes, dict[str, str]] | None:
        """Return tuple containing the object content and metadata."""
        with self._lock:
            entry = self._store.get(key)

        if entry is None:
            return None

        content, verifications_json = entry
        if hashlib.sha256(content).hexdigest() != key:
            log(
                ERROR,
                "FAB hash mismatch for key '%s' when reading from in-memory store.",
                key,
            )
            return None

        meta = _from_verifications_json(verifications_json)
        if meta is None:
            log(
                ERROR,
                "Corrupted FAB verifications for key '%s' in in-memory store.",
                key,
            )
            return None

        return content, meta

    def delete(self, key: str) -> None:
        """Delete object with hash."""
        with self._lock:
            self._store.pop(key, None)

    def list(self) -> list[str]:
        """List all keys."""
        with self._lock:
            return list(self._store.keys())


class _SqlObjectStoreFfs(Ffs, SqlMixin):
    """SQL-backed FAB storage implementation."""

    def __init__(self, database_path: str) -> None:
        super().__init__(database_path)
        self.initialize()

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData for FAB storage tables."""
        return create_objectstore_metadata()

    def put(self, content: bytes, meta: dict[str, str]) -> str:
        """Store bytes and metadata and return key (hash of content)."""
        content_hash = hashlib.sha256(content).hexdigest()
        verifications_json = _to_verifications_json(meta)

        with self.session():
            rows = self.query(
                "INSERT INTO fab_objects (fab_hash, content, verifications_json) "
                "VALUES (:fab_hash, :content, :verifications_json) "
                "ON CONFLICT (fab_hash) DO NOTHING "
                "RETURNING fab_hash",
                {
                    "fab_hash": content_hash,
                    "content": content,
                    "verifications_json": verifications_json,
                },
            )

            if rows:
                return content_hash

            existing_rows = self.query(
                "SELECT content, verifications_json FROM fab_objects "
                "WHERE fab_hash = :fab_hash",
                {"fab_hash": content_hash},
            )

        if not existing_rows:
            raise ValueError(
                f"FAB row for hash '{content_hash}' disappeared during put operation."
            )

        existing_content = existing_rows[0]["content"]
        existing_meta = _from_verifications_json(existing_rows[0]["verifications_json"])
        if existing_content != content or existing_meta != meta:
            raise ValueError(
                "Conflicting FAB content or verification metadata detected "
                f"for hash '{content_hash}'."
            )

        return content_hash

    def get(self, key: str) -> tuple[bytes, dict[str, str]] | None:
        """Return tuple containing the object content and metadata."""
        rows = self.query(
            "SELECT content, verifications_json FROM fab_objects "
            "WHERE fab_hash = :fab_hash",
            {"fab_hash": key},
        )

        if not rows:
            return None

        content = rows[0]["content"]
        if hashlib.sha256(content).hexdigest() != key:
            log(ERROR, "FAB hash mismatch for key '%s' in SQL store.", key)
            return None

        meta = _from_verifications_json(rows[0]["verifications_json"])
        if meta is None:
            log(ERROR, "Corrupted FAB verifications for key '%s' in SQL store.", key)
            return None

        return content, meta

    def delete(self, key: str) -> None:
        """Delete object with hash."""
        self.query(
            "DELETE FROM fab_objects WHERE fab_hash = :fab_hash",
            {"fab_hash": key},
        )

    def list(self) -> list[str]:
        """List all keys."""
        rows = self.query("SELECT fab_hash FROM fab_objects ORDER BY fab_hash")
        return [str(row["fab_hash"]) for row in rows]


class ObjectStoreFfs(Ffs):
    """Ffs implementation backed by in-memory or SQL FAB storage."""

    def __init__(self, database: str = FLWR_IN_MEMORY_DB_NAME) -> None:
        if database == FLWR_IN_MEMORY_DB_NAME:
            log(DEBUG, "Initializing in-memory ObjectStoreFfs")
            self._impl: Ffs = _InMemoryObjectStoreFfs()
            return

        log(DEBUG, "Initializing SQL ObjectStoreFfs")
        self._impl = _SqlObjectStoreFfs(database)

    def put(self, content: bytes, meta: dict[str, str]) -> str:
        """Store bytes and metadata and return key (hash of content)."""
        return self._impl.put(content, meta)

    def get(self, key: str) -> tuple[bytes, dict[str, str]] | None:
        """Return tuple containing the object content and metadata."""
        return self._impl.get(key)

    def delete(self, key: str) -> None:
        """Delete object with hash."""
        self._impl.delete(key)

    def list(self) -> list[str]:
        """List all keys."""
        return self._impl.list()

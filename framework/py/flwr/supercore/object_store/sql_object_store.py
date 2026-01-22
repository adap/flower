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
"""Flower SQLAlchemy-based ObjectStore implementation."""


from sqlalchemy import MetaData

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

from .object_store import ObjectStore


class SqlObjectStore(ObjectStore, SqlMixin):
    """SQLAlchemy-based implementation of the ObjectStore interface."""

    def __init__(self, database_path: str, verify: bool = True) -> None:
        super().__init__(database_path)
        self.verify = verify

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData for ObjectStore tables."""
        return create_objectstore_metadata()

    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`."""
        raise NotImplementedError()

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        raise NotImplementedError()

    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store."""
        raise NotImplementedError()

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        rows = self.query(
            "SELECT content FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        return rows[0]["content"] if rows else None

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        raise NotImplementedError()

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        raise NotImplementedError()

    def clear(self) -> None:
        """Clear the store."""
        with self.session():
            self.query("DELETE FROM object_children")
            self.query("DELETE FROM run_objects")
            self.query("DELETE FROM objects")

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        rows = self.query(
            "SELECT 1 FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        return len(rows) > 0

    def __len__(self) -> int:
        """Return the number of objects in the store."""
        rows = self.query("SELECT COUNT(*) AS cnt FROM objects")
        return int(rows[0]["cnt"])

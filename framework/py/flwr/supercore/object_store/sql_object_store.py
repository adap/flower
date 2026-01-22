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

from flwr.common.inflatable import (
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)
from flwr.common.inflatable_utils import validate_object_content
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata
from flwr.supercore.utils import uint64_to_int64

from .object_store import NoObjectInStoreError, ObjectStore


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
        new_objects = []
        with self.session():
            for tree_node in iterate_object_tree(object_tree):
                obj_id = tree_node.object_id
                if not is_valid_sha256_hash(obj_id):
                    raise ValueError(f"Invalid object ID format: {obj_id}")

                child_ids = [child.object_id for child in tree_node.children]

                # Check if object exists
                rows = self.query(
                    "SELECT object_id, is_available FROM objects "
                    "WHERE object_id = :oid",
                    {"oid": obj_id},
                )

                if not rows:
                    # Insert new object
                    self.query(
                        "INSERT INTO objects "
                        "(object_id, content, is_available, ref_count) "
                        "VALUES (:oid, :content, :avail, :ref)",
                        {"oid": obj_id, "content": b"", "avail": 0, "ref": 0},
                    )
                    # Insert child relationships and increment ref counts
                    for cid in child_ids:
                        self.query(
                            "INSERT INTO object_children (parent_id, child_id) "
                            "VALUES (:pid, :cid)",
                            {"pid": obj_id, "cid": cid},
                        )
                        self.query(
                            "UPDATE objects SET ref_count = ref_count + 1 "
                            "WHERE object_id = :cid",
                            {"cid": cid},
                        )
                    new_objects.append(obj_id)
                else:
                    # Add to list if not available
                    if not rows[0]["is_available"]:
                        new_objects.append(obj_id)

                # Ensure run mapping (idempotent)
                self.query(
                    "INSERT INTO run_objects (run_id, object_id) "
                    "VALUES (:rid, :oid) ON CONFLICT DO NOTHING",
                    {"rid": uint64_to_int64(run_id), "oid": obj_id},
                )

        return new_objects

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        # Verify object exists
        rows = self.query(
            "SELECT object_id FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        if not rows:
            raise NoObjectInStoreError(f"Object {object_id} was not pre-registered.")

        # Get children
        children = self.query(
            "SELECT child_id FROM object_children WHERE parent_id = :oid",
            {"oid": object_id},
        )

        # Build the object trees of all children recursively
        try:
            child_trees = [self.get_object_tree(ch["child_id"]) for ch in children]
        except NoObjectInStoreError as e:
            # Raise an error if any child object is missing
            raise NoObjectInStoreError(
                f"Object tree for object ID '{object_id}' contains missing "
                "children. This may indicate a corrupted object store."
            ) from e

        # Create and return the ObjectTree for the current object
        return ObjectTree(object_id=object_id, children=child_trees)

    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store."""
        if self.verify:
            # Verify object_id and object_content match
            object_id_from_content = get_object_id(object_content)
            if object_id != object_id_from_content:
                raise ValueError(f"Object ID {object_id} does not match content hash")

            # Validate object content
            validate_object_content(content=object_content)

        # Only allow adding the object if it has been preregistered
        rows = self.query(
            "SELECT is_available FROM objects WHERE object_id = :oid",
            {"oid": object_id},
        )
        if not rows:
            raise NoObjectInStoreError(
                f"Object with ID '{object_id}' was not pre-registered."
            )

        # Return if object is already present in the store
        if rows[0]["is_available"]:
            return

        # Update the object entry in the store
        self.query(
            "UPDATE objects SET content = :content, is_available = 1 "
            "WHERE object_id = :oid",
            {"content": object_content, "oid": object_id},
        )

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

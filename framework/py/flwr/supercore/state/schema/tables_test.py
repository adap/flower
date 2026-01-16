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
"""Tests to verify schema parity between tables.py and sqlite_linkstate.py.

This is a temporary test to verify schema parity and will be removed after the
SQLAlchemy-based LinkState is fully implemented.
"""

import unittest

from sqlalchemy import create_engine, inspect, text

from flwr.supercore.corestate.sqlite_corestate import SQL_CREATE_TABLE_TOKEN_STORE
from flwr.supercore.state.schema.corestate_tables import corestate_metadata


class SchemaParityTest(unittest.TestCase):
    """Test that SQLAlchemy schema matches raw SQL schema."""

    def setUp(self) -> None:
        """Set up test databases."""
        # Create database with raw SQL (the "expected" schema)
        self.raw_engine = create_engine("sqlite:///:memory:")
        with self.raw_engine.connect() as conn:
            # CoreState tables
            conn.execute(text(SQL_CREATE_TABLE_TOKEN_STORE))
            conn.commit()

        # Create database with SQLAlchemy metadata (the "actual" schema)
        self.sqlalchemy_engine = create_engine("sqlite:///:memory:")
        corestate_metadata.create_all(self.sqlalchemy_engine)

        # Cache inspectors for use in all tests
        self.raw_inspector = inspect(self.raw_engine)
        self.sqla_inspector = inspect(self.sqlalchemy_engine)

    def test_table_names_match(self) -> None:
        """Verify both schemas have the same table names."""
        # Prepare
        raw_tables = set(self.raw_inspector.get_table_names())
        sqla_tables = set(self.sqla_inspector.get_table_names())

        # Assert
        self.assertEqual(raw_tables, sqla_tables, "Table names do not match")

    def test_column_names_match(self) -> None:
        """Verify each table has the same column names."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare
            raw_cols = {c["name"] for c in self.raw_inspector.get_columns(table_name)}
            sqla_cols = {c["name"] for c in self.sqla_inspector.get_columns(table_name)}

            # Assert
            self.assertEqual(
                raw_cols,
                sqla_cols,
                f"Column names do not match for table '{table_name}'",
            )

    def test_column_order_matches(self) -> None:
        """Verify columns are in the same order (important for INSERT statements)."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare
            raw_cols = [c["name"] for c in self.raw_inspector.get_columns(table_name)]
            sqla_cols = [c["name"] for c in self.sqla_inspector.get_columns(table_name)]

            # Assert
            self.assertEqual(
                raw_cols,
                sqla_cols,
                f"Column order does not match for table '{table_name}'",
            )

    def test_index_names_match(self) -> None:
        """Verify both schemas have the same indexes."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare
            raw_indexes = {
                idx["name"] for idx in self.raw_inspector.get_indexes(table_name)
            }
            sqla_indexes = {
                idx["name"] for idx in self.sqla_inspector.get_indexes(table_name)
            }

            # Assert
            self.assertEqual(
                raw_indexes,
                sqla_indexes,
                f"Index names do not match for table '{table_name}'",
            )

    def test_unique_constraints_match(self) -> None:
        """Verify unique constraints are the same."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare: Get unique constraints from indexes (SQLite reports them as
            # unique indexes)
            raw_unique_cols = {
                tuple(idx["column_names"])
                for idx in self.raw_inspector.get_indexes(table_name)
                if idx["unique"]
            }
            sqla_unique_cols = {
                tuple(idx["column_names"])
                for idx in self.sqla_inspector.get_indexes(table_name)
                if idx["unique"]
            }

            # Assert
            self.assertEqual(
                raw_unique_cols,
                sqla_unique_cols,
                f"Unique constraints do not match for table '{table_name}'",
            )

    def test_column_types_match(self) -> None:
        """Verify column types are compatible between schemas.

        Note: SQLite type affinity means types are loosely matched. We compare
        the type strings after normalizing common equivalences.
        """
        raw_types_seen: set[str] = set()

        # SQLite type affinity mapping: different spellings map to same affinity
        def normalize_type(type_str: str) -> str:
            """Normalize SQLite type names to their affinity class."""
            type_upper = str(type_str).upper()

            # Define affinity rules as (keywords, affinity) tuples
            # Order matters: check specific patterns before generic ones
            affinity_rules = [
                (["INT"], "INTEGER"),
                (["CHAR", "TEXT", "CLOB"], "TEXT"),
                (["REAL", "FLOA", "DOUB", "TIMESTAMP"], "REAL"),
                (["BLOB", "BINARY"], "BLOB"),
            ]

            # Check empty string special case
            if type_upper == "":
                return "BLOB"

            # Find matching affinity rule
            for keywords, affinity in affinity_rules:
                if any(keyword in type_upper for keyword in keywords):
                    return affinity

            # No match found, return as-is
            return type_upper

        # Assert: Check types for each table
        for table_name in self.raw_inspector.get_table_names():
            raw_cols = {
                c["name"]: normalize_type(str(c["type"]))
                for c in self.raw_inspector.get_columns(table_name)
            }
            sqla_cols = {
                c["name"]: normalize_type(str(c["type"]))
                for c in self.sqla_inspector.get_columns(table_name)
            }
            raw_types_seen.update(
                str(c["type"]).upper()
                for c in self.raw_inspector.get_columns(table_name)
            )

            for col_name in raw_cols:
                self.assertEqual(
                    raw_cols[col_name],
                    sqla_cols[col_name],
                    f"Type mismatch for column '{col_name}' in table '{table_name}': "
                    f"raw={raw_cols[col_name]}, sqla={sqla_cols[col_name]}",
                )

    def test_foreign_keys_match(self) -> None:
        """Verify foreign key constraints are the same."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare
            raw_fks = {
                (fk["referred_table"], tuple(fk["constrained_columns"]))
                for fk in self.raw_inspector.get_foreign_keys(table_name)
            }
            sqla_fks = {
                (fk["referred_table"], tuple(fk["constrained_columns"]))
                for fk in self.sqla_inspector.get_foreign_keys(table_name)
            }

            # Assert
            self.assertEqual(
                raw_fks,
                sqla_fks,
                f"Foreign keys do not match for table '{table_name}'",
            )

    def test_nullable_constraints_match(self) -> None:
        """Verify nullable constraints are the same."""
        for table_name in self.raw_inspector.get_table_names():
            # Prepare
            raw_nullable = {
                c["name"]: c["nullable"]
                for c in self.raw_inspector.get_columns(table_name)
            }
            sqla_nullable = {
                c["name"]: c["nullable"]
                for c in self.sqla_inspector.get_columns(table_name)
            }

            # Assert
            for col_name in raw_nullable:
                self.assertEqual(
                    raw_nullable[col_name],
                    sqla_nullable[col_name],
                    f"Nullable mismatch for column '{col_name}' in "
                    f"table '{table_name}': raw={raw_nullable[col_name]}, "
                    f"sqla={sqla_nullable[col_name]}",
                )


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
"""Tests for SqliteMixin and SqlMixin."""


import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from parameterized import parameterized
from sqlalchemy import Column, Integer, MetaData, Table, text

from .sql_mixin import SqlMixin
from .sqlite_mixin import SqliteMixin


class DummyDb(SqliteMixin):
    """Simple subclass for testing SqliteMixin behavior."""

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements for test table."""
        return (
            "CREATE TABLE IF NOT EXISTS test"
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER)",
        )


class DummyDbSqlAlchemy(SqlMixin):
    """Simple subclass for testing SqlMixin behavior with SQLAlchemy."""

    def get_metadata(self) -> MetaData:
        """Return MetaData with test table definition."""
        metadata = MetaData()
        Table(
            "test",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("value", Integer),
        )
        return metadata


@parameterized.expand(
    [
        (DummyDb,),
        (DummyDbSqlAlchemy,),
    ],
    ids=["SqliteMixin", "SqlMixin"],
)  # type: ignore
def test_transaction_serialization_with_tempfile(
    db_class: type[DummyDb] | type[DummyDbSqlAlchemy],
) -> None:
    """Verify that SQLite file-level locking serializes concurrent transactions."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpfile:
        db_path = tmpfile.name

    try:
        # Initialize database schema once
        init_db = db_class(db_path)
        init_db.initialize()

        def insert_row(_: int) -> None:
            # Each thread creates its own connection to test file-level locking
            db = db_class(db_path)
            db.initialize()
            if isinstance(db, DummyDb):
                # SqliteMixin: use conn context and ? placeholders
                with db.conn as conn:
                    # Insert a dummy row with value -1
                    conn.execute("INSERT INTO test (value) VALUES (?)", (-1,))
                    # Read current row count
                    count = db.conn.execute(
                        "SELECT COUNT(*) AS cnt FROM test"
                    ).fetchone()["cnt"]
                    # Simulate some processing time
                    time.sleep(0.001)
                    # Insert a new row with the current count
                    conn.execute("INSERT INTO test (value) VALUES (?)", (count,))
            else:
                # SqlMixin: use session context for single atomic transaction
                with db.session() as session:
                    # Insert a dummy row with value -1
                    session.execute(
                        text("INSERT INTO test (value) VALUES (:value)"), {"value": -1}
                    )
                    # Read current row count
                    result = session.execute(text("SELECT COUNT(*) AS cnt FROM test"))
                    row = result.mappings().fetchone()
                    assert row is not None
                    count = row["cnt"]
                    # Simulate some processing time
                    time.sleep(0.001)
                    # Insert a new row with the current count
                    session.execute(
                        text("INSERT INTO test (value) VALUES (:value)"),
                        {"value": count},
                    )
                    session.commit()

        # Execute: Run concurrent transactions
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(insert_row, range(100))

        # Assert: Verify that all rows were inserted correctly
        rows = init_db.query("SELECT * FROM test ORDER BY id")
        for row in rows:
            if row["id"] & 0x1:
                # Odd IDs are dummy rows
                assert row["value"] == -1
            else:
                # Even IDs should have sequential counts
                assert row["value"] == row["id"] - 1

    finally:
        # Clean up the temporary file
        os.unlink(db_path)


def test_sql_mixin_session_reuse() -> None:
    """Test that nested query() calls reuse the same session."""
    db = DummyDbSqlAlchemy(":memory:")
    db.initialize()

    # Insert initial test data
    db.query("INSERT INTO test (value) VALUES (:value)", {"value": 100})

    # Test: Multiple query() calls within a session should share the same transaction
    with db.session() as session:
        # First query: insert a value
        db.query("INSERT INTO test (value) VALUES (:value)", {"value": 200})

        # Second query: verify the value was inserted
        rows = db.query("SELECT value FROM test WHERE value = :value", {"value": 200})
        assert len(rows) == 1
        assert rows[0]["value"] == 200

        # Third query: update the value
        db.query(
            "UPDATE test SET value = :new WHERE value = :old", {"old": 200, "new": 300}
        )

        # Fourth query: verify the update
        rows = db.query("SELECT value FROM test WHERE value = :value", {"value": 300})
        assert len(rows) == 1
        assert rows[0]["value"] == 300

    # Verify all changes were committed
    rows = db.query("SELECT value FROM test ORDER BY value")
    assert len(rows) == 2
    assert rows[0]["value"] == 100
    assert rows[1]["value"] == 300


def test_sql_mixin_session_rollback() -> None:
    """Test that exceptions in a session cause rollback for all nested queries."""
    db = DummyDbSqlAlchemy(":memory:")
    db.initialize()

    # Insert initial test data
    db.query("INSERT INTO test (value) VALUES (:value)", {"value": 100})

    # Test: Exception should rollback all nested query() calls
    try:
        with db.session():
            # First query: insert a value
            db.query("INSERT INTO test (value) VALUES (:value)", {"value": 200})

            # Second query: verify the value was inserted (within transaction)
            rows = db.query(
                "SELECT value FROM test WHERE value = :value", {"value": 200}
            )
            assert len(rows) == 1

            # Raise an exception to trigger rollback
            raise ValueError("Simulated error")
    except ValueError:
        pass  # Expected

    # Verify the transaction was rolled back
    rows = db.query("SELECT value FROM test")
    assert len(rows) == 1
    assert rows[0]["value"] == 100


def test_sql_mixin_nested_sessions() -> None:
    """Test that nested session() calls reuse the same session."""
    db = DummyDbSqlAlchemy(":memory:")
    db.initialize()

    # Test: Nested session contexts should reuse the same session
    with db.session() as outer_session:
        db.query("INSERT INTO test (value) VALUES (:value)", {"value": 100})

        with db.session() as inner_session:
            # Inner session should be the same object as outer session
            assert inner_session is outer_session

            # Insert in nested context
            db.query("INSERT INTO test (value) VALUES (:value)", {"value": 200})

        # After inner context, can still use outer session
        db.query("INSERT INTO test (value) VALUES (:value)", {"value": 300})

    # Verify all inserts were committed as one transaction
    rows = db.query("SELECT value FROM test ORDER BY value")
    assert len(rows) == 3
    assert rows[0]["value"] == 100
    assert rows[1]["value"] == 200
    assert rows[2]["value"] == 300


def test_sql_mixin_query_without_session() -> None:
    """Test that query() works independently when not in a session context."""
    db = DummyDbSqlAlchemy(":memory:")
    db.initialize()

    # Each query() call should be its own transaction
    db.query("INSERT INTO test (value) VALUES (:value)", {"value": 100})
    db.query("INSERT INTO test (value) VALUES (:value)", {"value": 200})
    db.query("INSERT INTO test (value) VALUES (:value)", {"value": 300})

    # Verify all inserts were committed independently
    rows = db.query("SELECT value FROM test ORDER BY value")
    assert len(rows) == 3
    assert rows[0]["value"] == 100
    assert rows[1]["value"] == 200
    assert rows[2]["value"] == 300

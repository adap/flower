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
"""Tests for SqliteMixin."""


import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from parameterized import parameterized

from .sqlite_mixin import SqliteMixin


class DummyDb(SqliteMixin):
    """Simple subclass for testing SqliteMixin behavior."""

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements for test table."""
        return (
            "CREATE TABLE IF NOT EXISTS test"
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER)",
        )


@parameterized.expand(
    [
        (DummyDb,),
    ],
    ids=["SqliteMixin"],
)  # type: ignore
def test_transaction_serialization_with_tempfile(
    db_class: type[DummyDb],
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
            # SqliteMixin: use conn context and ? placeholders
            with db.conn:
                # Insert a dummy row with value -1
                db.conn.execute("INSERT INTO test (value) VALUES (?)", (-1,))
                # Read current row count
                count = db.conn.execute("SELECT COUNT(*) AS cnt FROM test").fetchone()[
                    "cnt"
                ]
                # Simulate some processing time
                time.sleep(0.001)
                # Insert a new row with the current count
                db.conn.execute("INSERT INTO test (value) VALUES (?)", (count,))

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

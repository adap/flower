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


import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from .sqlite_mixin import SqliteMixin


class DummyDb(SqliteMixin):
    """Simple subclass for testing SqliteMixin behavior."""

    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Initialize the database with a simple test table."""
        return self._ensure_initialized(
            "CREATE TABLE IF NOT EXISTS test"
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER)"
        )


def test_transaction_serialization_with_tempfile() -> None:
    """Verify that `.conn` runs inside real SQLite transactions."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmpfile:
        # Prepare:
        def insert_row(_: int) -> None:
            db = DummyDb(tmpfile.name)
            db.initialize()
            with db.conn:
                # Insert a dummy row with value -1
                db.conn.execute("INSERT INTO test (value) VALUES (?)", (-1,))
                with db.conn:
                    # Read current row count
                    count = db.conn.execute(
                        "SELECT COUNT(*) AS cnt FROM test"
                    ).fetchone()["cnt"]
                    # Simulate some processing time
                    time.sleep(0.001)
                    # Insert a new row with the current count
                    db.conn.execute("INSERT INTO test (value) VALUES (?)", (count,))

        # Execute: Run concurrent transactions
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(insert_row, range(100))

        # Assert: Verify that all rows were inserted correctly
        db = DummyDb(tmpfile.name)
        db.initialize()
        rows = db.query("SELECT * FROM test")
        for row in rows:
            if row["id"] & 0x1:
                # Odd IDs are dummy rows
                assert row["value"] == -1
            else:
                # Even IDs should have sequential counts
                assert row["value"] == row["id"] - 1

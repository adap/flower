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
"""SQLite based implemenation of SuperExec state."""


import sqlite3
from typing import Optional

from typing_extensions import override

from .state import ExecState, RunStatus


class SqliteExecState(ExecState):
    """SQLite implementation of ExecState."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY,
                    status INTEGER
                )
            """
            )

    @override
    def update_run_status(self, run_id: int, status: RunStatus) -> None:
        """Store or update a RunStatus in the database."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
            if cursor.fetchone():
                self.conn.execute(
                    "UPDATE runs SET status = ? WHERE run_id = ?",
                    (status.value, run_id),
                )
            else:
                self.conn.execute(
                    "INSERT INTO runs (run_id, status) VALUES (?, ?)",
                    (run_id, status.value),
                )

    @override
    def get_run_status(self, run_id: int) -> Optional[RunStatus]:
        """Get a RunStatus from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return RunStatus(row[0]) if row else None

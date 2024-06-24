# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""SQLite based implemenation of server state."""


import datetime
from typing import List, Optional

import sqlite3
from typing_extensions import override

from .state import RunStatus, SuperexecState


class SqliteSuperexecState(SuperexecState):
    """SQLite implementation of SuperexecState."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    @override
    def initialize(self):
        """Initialize the database."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY,
                    status INTEGER
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    timestamp TEXT,
                    stream TEXT,
                    message TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """
            )

    @override
    def store_log(self, run_id: int, log_output: str, stream: str = "stderr") -> None:
        """Store logs into the database."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO logs (run_id, timestamp, stream, message) "
                "VALUES (?, ?, ?, ?)",
                (run_id, datetime.datetime.now().isoformat(), stream, log_output),
            )

    @override
    def get_logs(self, run_id: int) -> List[str]:
        """Get logs from the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT log_output FROM logs WHERE run_id = ? ORDER BY id ASC", (run_id,)
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    @override
    def update_run_tracker(self, run_id: int, status: RunStatus) -> None:
        """Store or update a RunTracker in the database."""
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
    def get_run_tracker_status(self, run_id: int) -> Optional[RunStatus]:
        """Get a RunTracker's status from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return RunStatus(row[0]) if row else None

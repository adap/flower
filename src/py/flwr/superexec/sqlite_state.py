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


import json
import sqlite3
from typing import Optional

from typing_extensions import override

from .state import ExecState
from flwr.common.typing import UserConfig


class SqliteExecState(ExecState):
    """SQLite implementation of ExecState."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY,
                    run_config TEXT,
                    fab_hash TEXT
                )
            """
            )

    @override
    def store_run(self, run_id: int, run_config: UserConfig, fab_hash: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, run_config, fab_hash) VALUES (?, ?, ?)",
                (run_id, json.dumps(run_config), fab_hash),
            )

    @override
    def get_run_config(self, run_id: int) -> Optional[UserConfig]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT run_config FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return UserConfig(json.loads(row[0])) if row else None

    @override
    def get_fab_hash(self, run_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT fab_hash FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    @override
    def get_runs(self) -> list[int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM runs WHERE run_id = ")
        row = cursor.fetchone()
        return [int(row[0])] if row else []

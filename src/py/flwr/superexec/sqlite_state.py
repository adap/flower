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

from flwr.common.typing import UserConfig
from flwr.server.superlink.state.utils import (
    convert_uint64_to_sint64,
    convert_sint64_to_uint64,
)

from .state import ExecState


class SqliteExecState(ExecState):
    """SQLite implementation of ExecState."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY,
                run_config TEXT,
                fab_hash TEXT
            );
        """
        )

    @override
    def store_run(self, run_id: int, run_config: UserConfig, fab_hash: str) -> None:
        if self.conn is None:
            raise AttributeError("State is not initialized.")

        self.conn.execute(
            "INSERT INTO runs (run_id, run_config, fab_hash) VALUES (?, ?, ?)",
            (convert_uint64_to_sint64(run_id), json.dumps(run_config), fab_hash),
        )

    @override
    def get_run_config(self, run_id: int) -> Optional[UserConfig]:
        if self.conn is None:
            raise AttributeError("State is not initialized.")

        with self.conn:
            res = self.conn.execute(
                "SELECT run_config FROM runs WHERE run_id = ?", (run_id,)
            )
            row = res.fetchone()
        return UserConfig(json.loads(row[0])) if row else None

    @override
    def get_fab_hash(self, run_id: int) -> Optional[str]:
        if self.conn is None:
            raise AttributeError("State is not initialized.")

        with self.conn:
            res = self.conn.execute(
                "SELECT fab_hash FROM runs WHERE run_id = ?", (run_id,)
            )
        row = res.fetchone()
        return row[0] if row else None

    @override
    def get_runs(self) -> list[int]:
        if self.conn is None:
            raise AttributeError("State is not initialized.")

        with self.conn:
            res = self.conn.execute("SELECT run_id FROM runs")
        rows = res.fetchall()
        return [convert_sint64_to_uint64(int(row[0])) for row in rows]

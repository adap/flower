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
"""SQLite-based CoreState implementation."""


import secrets
import sqlite3
from typing import cast

from flwr.common.constant import FLWR_APP_TOKEN_LENGTH
from flwr.supercore.sqlite_mixin import SqliteMixin
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64

from .corestate import CoreState

SQL_CREATE_TABLE_TOKEN_STORE = """
CREATE TABLE IF NOT EXISTS token_store (
    run_id                  INTEGER PRIMARY KEY,
    token                   TEXT UNIQUE NOT NULL
);
"""


class SqliteCoreState(CoreState, SqliteMixin):
    """SQLite-based CoreState implementation."""

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements needed for CoreState tables."""
        return (SQL_CREATE_TABLE_TOKEN_STORE,)

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        query = "INSERT INTO token_store (run_id, token) VALUES (:run_id, :token);"
        data = {"run_id": uint64_to_int64(run_id), "token": token}
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            return None  # Token already created for this run ID
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        query = "SELECT token FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        rows = self.query(query, data)
        if not rows:
            return False
        return cast(str, rows[0]["token"]) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        query = "DELETE FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        self.query(query, data)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        query = "SELECT run_id FROM token_store WHERE token = :token;"
        data = {"token": token}
        rows = self.query(query, data)
        if not rows:
            return None
        return int64_to_uint64(rows[0]["run_id"])

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        raise NotImplementedError

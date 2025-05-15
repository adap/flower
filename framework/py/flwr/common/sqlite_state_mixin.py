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
"""Mixin for adding SQLite state management functionality to classes."""


import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from logging import DEBUG, ERROR
from typing import Any, Optional, Union

from flwr.common import log

DictOrTuple = Union[tuple[Any, ...], dict[str, Any]]


class SqliteStateMixin(ABC):  # pylint: disable=R0904
    """SQLite-based state management mixin."""

    def __init__(
        self,
        database_path: str,
    ) -> None:
        """Initialize an SqliteStateMixin.

        Parameters
        ----------
        database : (path-like object)
            The path to the database file to be opened. Pass ":memory:" to open
            a connection to a database that is in RAM, instead of on disk.
        """
        self.database_path = database_path
        self.conn: Optional[sqlite3.Connection] = None

    @property
    @abstractmethod
    def schema_setup_commands(self) -> list[str]:
        """Return the schema setup commands."""

    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Create tables if they don't exist yet.

        Parameters
        ----------
        log_queries : bool
            Log each query which is executed.

        Returns
        -------
        list[tuple[str]]
            The list of all tables in the DB.
        """
        self.conn = sqlite3.connect(self.database_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.row_factory = dict_factory
        if log_queries:
            self.conn.set_trace_callback(lambda query: log(DEBUG, query))
        cur = self.conn.cursor()

        # Create each table if not exists queries
        for cmd in self.schema_setup_commands:
            cur.execute(cmd)
        res = cur.execute("SELECT name FROM sqlite_schema;")
        return res.fetchall()

    def query(
        self,
        query: str,
        data: Optional[Union[Sequence[DictOrTuple], DictOrTuple]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query."""
        if self.conn is None:
            raise AttributeError("LinkState is not initialized.")

        if data is None:
            data = []

        # Clean up whitespace to make the logs nicer
        query = re.sub(r"\s+", " ", query)

        try:
            with self.conn:
                if (
                    len(data) > 0
                    and isinstance(data, (tuple, list))
                    and isinstance(data[0], (tuple, dict))
                ):
                    rows = self.conn.executemany(query, data)
                else:
                    rows = self.conn.execute(query, data)

                # Extract results before committing to support
                #   INSERT/UPDATE ... RETURNING
                # style queries
                result = rows.fetchall()
        except KeyError as exc:
            log(ERROR, {"query": query, "data": data, "exception": exc})

        return result


def dict_factory(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
) -> dict[str, Any]:
    """Turn SQLite results into dicts.

    Less efficent for retrival of large amounts of data but easier to use.
    """
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))

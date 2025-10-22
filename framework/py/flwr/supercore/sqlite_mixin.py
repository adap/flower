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
"""Mixin providing common SQLite connection and initialization logic."""


import contextlib
import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from logging import DEBUG, ERROR
from typing import Any, Optional, Union

from flwr.common.logger import log

DictOrTuple = Union[tuple[Any, ...], dict[str, Any]]


class SqliteMixin(ABC):
    """Mixin providing common SQLite connection and initialization logic."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the SQLite connection."""
        if self._conn is None:
            raise AttributeError("Database not initialized. Call initialize() first.")
        return self._conn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for a transaction.

        This allows nesting of transactions by checking if a transaction is
        already in progress.

        Examples
        --------
        ::

            with self.transaction():
                # Do some DB operations here
                ...
                with self.transaction():
                    # Do some more DB operations here
                    ...
        """
        if self._conn is None:
            raise AttributeError("Database not initialized. Call initialize() first.")

        # Start a transaction if not already in one
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN")
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        # Do nothing if already in a transaction
        else:
            yield

    @abstractmethod
    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Connect to the DB, enable FK support, and create tables if needed.

        Parameters
        ----------
        log_queries : bool
            Log each query which is executed.

        Returns
        -------
        list[tuple[str]]
            The list of all tables in the DB.

        Examples
        --------
        Implement in subclass:

        .. code:: python

            def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
                return self._ensure_initialized(
                    SQL_CREATE_TABLE_FOO,
                    SQL_CREATE_TABLE_BAR,
                    log_queries=log_queries
                )
        """

    def _ensure_initialized(
        self,
        *create_statements: str,
        log_queries: bool = False,
    ) -> list[tuple[str]]:
        """Connect to the DB, enable FK support, and create tables if needed.

        Subclasses should call this with their own CREATE TABLE/INDEX statements in
        their `.initialize()` methods.

        Parameters
        ----------
        create_statements : str
            SQL statements to create tables and indexes.
        log_queries : bool
            Log each query which is executed.

        Returns
        -------
        list[tuple[str]]
            The list of all tables in the DB.
        """
        self._conn = sqlite3.connect(self.database_path)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = dict_factory

        if log_queries:
            self._conn.set_trace_callback(lambda q: log(DEBUG, q))

        # Create tables and indexes
        cur = self._conn.cursor()
        for sql in create_statements:
            cur.execute(sql)
        res = cur.execute("SELECT name FROM sqlite_schema;")
        return res.fetchall()

    def query(
        self,
        query: str,
        data: Optional[Union[Sequence[DictOrTuple], DictOrTuple]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return the results as list of dicts."""
        if self._conn is None:
            raise AttributeError("LinkState is not initialized.")

        if data is None:
            data = []

        # Clean up whitespace to make the logs nicer
        query = re.sub(r"\s+", " ", query)

        try:
            with self.transaction():
                if (
                    len(data) > 0
                    and isinstance(data, (tuple, list))
                    and isinstance(data[0], (tuple, dict))
                ):
                    rows = self._conn.executemany(query, data)
                else:
                    rows = self._conn.execute(query, data)

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

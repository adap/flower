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


import os
import re
import sqlite3
import time
from logging import DEBUG, ERROR
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from uuid import UUID, uuid4

from flwr.common import log, now
from flwr.common.typing import Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.recordset_pb2 import RecordSet  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes  # pylint: disable=E0611
from flwr.server.utils.validator import validate_task_ins_or_res


SQL_CREATE_TABLE_NODE = """
CREATE TABLE IF NOT EXISTS node(
    node_id         INTEGER UNIQUE,
    online_until    REAL,
    ping_interval   REAL,
    public_key      BLOB
);
"""

SQL_CREATE_TABLE_CREDENTIAL = """
CREATE TABLE IF NOT EXISTS credential(
    private_key BLOB PRIMARY KEY,
    public_key BLOB
);
"""

SQL_CREATE_TABLE_PUBLIC_KEY = """
CREATE TABLE IF NOT EXISTS public_key(
    public_key BLOB UNIQUE
);
"""

SQL_CREATE_INDEX_ONLINE_UNTIL = """
CREATE INDEX IF NOT EXISTS idx_online_until ON node (online_until);
"""

SQL_CREATE_TABLE_RUN = """
CREATE TABLE IF NOT EXISTS run(
    run_id          INTEGER UNIQUE,
    fab_id          TEXT,
    fab_version     TEXT
);
"""

SQL_CREATE_TABLE_TASK_INS = """
CREATE TABLE IF NOT EXISTS task_ins(
    task_id                 TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    producer_anonymous      BOOLEAN,
    producer_node_id        INTEGER,
    consumer_anonymous      BOOLEAN,
    consumer_node_id        INTEGER,
    created_at              REAL,
    delivered_at            TEXT,
    pushed_at               REAL,
    ttl                     REAL,
    ancestry                TEXT,
    task_type               TEXT,
    recordset               BLOB,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_TASK_RES = """
CREATE TABLE IF NOT EXISTS task_res(
    task_id                 TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    producer_anonymous      BOOLEAN,
    producer_node_id        INTEGER,
    consumer_anonymous      BOOLEAN,
    consumer_node_id        INTEGER,
    created_at              REAL,
    delivered_at            TEXT,
    pushed_at               REAL,
    ttl                     REAL,
    ancestry                TEXT,
    task_type               TEXT,
    recordset               BLOB,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""

DictOrTuple = Union[Tuple[Any, ...], Dict[str, Any]]


class SqliteState:  # pylint: disable=R0904
    """SQLite-based state implementation."""

    def __init__(
        self,
        database_path: str,
    ) -> None:
        """Initialize an SqliteState.

        Parameters
        ----------
        database : (path-like object)
            The path to the database file to be opened. Pass ":memory:" to open
            a connection to a database that is in RAM, instead of on disk.
        """
        self.database_path = database_path
        self.conn: Optional[sqlite3.Connection] = None

    def initialize(self, log_queries: bool = False) -> List[Tuple[str]]:
        """Create tables if they don't exist yet.

        Parameters
        ----------
        log_queries : bool
            Log each query which is executed.
        """
        self.conn = sqlite3.connect(self.database_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.row_factory = dict_factory
        if log_queries:
            self.conn.set_trace_callback(lambda query: log(DEBUG, query))
        cur = self.conn.cursor()

        # Create each table if not exists queries
        cur.execute(SQL_CREATE_TABLE_RUN)
        cur.execute(SQL_CREATE_TABLE_NODE)
        res = cur.execute("SELECT name FROM sqlite_schema;")

        return res.fetchall()

    def query(
        self,
        query: str,
        data: Optional[Union[Sequence[DictOrTuple], DictOrTuple]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query."""
        if self.conn is None:
            raise AttributeError("State is not initialized.")

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

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns.

        Usually, the Driver API calls this to schedule instructions.

        Stores the value of the task_ins in the state and, if successful, returns the
        task_id (UUID) of the task_ins. If, for any reason, storing the task_ins fails,
        `None` is returned.

        Constraints
        -----------
        If `task_ins.task.consumer.anonymous` is `True`, then
        `task_ins.task.consumer.node_id` MUST NOT be set (equal 0).

        If `task_ins.task.consumer.anonymous` is `False`, then
        `task_ins.task.consumer.node_id` MUST be set (not 0)
        """
        # Validate task
        errors = validate_task_ins_or_res(task_ins)
        if any(errors):
            log(ERROR, errors)
            return None

        # Create task_id
        task_id = uuid4()

        # Store TaskIns
        task_ins.task_id = str(task_id)
        data = (task_ins_to_dict(task_ins),)
        columns = ", ".join([f":{key}" for key in data[0]])
        query = f"INSERT INTO task_ins VALUES({columns});"

        # Only invalid run_id can trigger IntegrityError.
        # This may need to be changed in the future version with more integrity checks.
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            log(ERROR, "`run` is invalid")
            return None

        return task_id

    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get undelivered TaskIns for one node (either anonymous or with ID).

        Usually, the Fleet API calls this for Nodes planning to work on one or more
        TaskIns.

        Constraints
        -----------
        If `node_id` is not `None`, retrieve all TaskIns where

            1. the `task_ins.task.consumer.node_id` equals `node_id` AND
            2. the `task_ins.task.consumer.anonymous` equals `False` AND
            3. the `task_ins.task.delivered_at` equals `""`.

        If `node_id` is `None`, retrieve all TaskIns where the
        `task_ins.task.consumer.node_id` equals `0` and
        `task_ins.task.consumer.anonymous` is set to `True`.

        `delivered_at` MUST BE set (i.e., not `""`) otherwise the TaskIns MUST not be in
        the result.

        If `limit` is not `None`, return, at most, `limit` number of `task_ins`. If
        `limit` is set, it has to be greater than zero.
        """
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        if node_id == 0:
            msg = (
                "`node_id` must be >= 1"
                "\n\n For requesting anonymous tasks use `node_id` equal `None`"
            )
            raise AssertionError(msg)

        data: Dict[str, Union[str, int]] = {}

        if node_id is None:
            # Retrieve all anonymous Tasks
            query = """
                SELECT task_id
                FROM task_ins
                WHERE consumer_anonymous == 1
                AND   consumer_node_id == 0
                AND   delivered_at = ""
            """
        else:
            # Retrieve all TaskIns for node_id
            query = """
                SELECT task_id
                FROM task_ins
                WHERE consumer_anonymous == 0
                AND   consumer_node_id == :node_id
                AND   delivered_at = ""
            """
            data["node_id"] = node_id

        if limit is not None:
            query += " LIMIT :limit"
            data["limit"] = limit

        query += ";"

        rows = self.query(query, data)

        if rows:
            # Prepare query
            task_ids = [row["task_id"] for row in rows]
            placeholders: str = ",".join([f":id_{i}" for i in range(len(task_ids))])
            query = f"""
                UPDATE task_ins
                SET delivered_at = :delivered_at
                WHERE task_id IN ({placeholders})
                RETURNING *;
            """

            # Prepare data for query
            delivered_at = now().isoformat()
            data = {"delivered_at": delivered_at}
            for index, task_id in enumerate(task_ids):
                data[f"id_{index}"] = str(task_id)

            # Run query
            rows = self.query(query, data)

        result = [dict_to_task_ins(row) for row in rows]

        return result

    def delete_tasks(self, task_ids: Set[UUID]) -> None:
        """Delete all delivered TaskIns/TaskRes pairs."""
        ids = list(task_ids)
        if len(ids) == 0:
            return None

        placeholders = ",".join([f":id_{index}" for index in range(len(task_ids))])
        data = {f"id_{index}": str(task_id) for index, task_id in enumerate(task_ids)}

        # 1. Query: Delete task_ins which have a delivered task_res
        query_1 = f"""
            DELETE FROM task_ins
            WHERE delivered_at != ''
            AND task_id IN (
                SELECT ancestry
                FROM task_res
                WHERE ancestry IN ({placeholders})
                AND delivered_at != ''
            );
        """

        # 2. Query: Delete delivered task_res to be run after 1. Query
        query_2 = f"""
            DELETE FROM task_res
            WHERE ancestry IN ({placeholders})
            AND delivered_at != '';
        """

        if self.conn is None:
            raise AttributeError("State not intitialized")

        with self.conn:
            self.conn.execute(query_1, data)
            self.conn.execute(query_2, data)

        return None

    def create_node(
        self, ping_interval: float, public_key: Optional[bytes] = None
    ) -> int:
        """Create, store in state, and return `node_id`."""
        # Sample a random int64 as node_id
        node_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

        query = "SELECT node_id FROM node WHERE public_key = :public_key;"
        row = self.query(query, {"public_key": public_key})

        if len(row) > 0:
            log(ERROR, "Unexpected node registration failure.")
            return 0

        query = (
            "INSERT INTO node "
            "(node_id, online_until, ping_interval, public_key) "
            "VALUES (?, ?, ?, ?)"
        )

        try:
            self.query(
                query, (node_id, time.time() + ping_interval, ping_interval, public_key)
            )
        except sqlite3.IntegrityError:
            log(ERROR, "Unexpected node registration failure.")
            return 0
        return node_id

    def delete_node(self, node_id: int, public_key: Optional[bytes] = None) -> None:
        """Delete a client node."""
        query = "DELETE FROM node WHERE node_id = ?"
        params = (node_id,)

        if public_key is not None:
            query += " AND public_key = ?"
            params += (public_key,)  # type: ignore

        if self.conn is None:
            raise AttributeError("State is not initialized.")

        try:
            with self.conn:
                rows = self.conn.execute(query, params)
                if rows.rowcount < 1:
                    raise ValueError("Public key or node_id not found")
        except KeyError as exc:
            log(ERROR, {"query": query, "data": params, "exception": exc})

    def get_nodes(self, run_id: int) -> Set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        # Validate run ID
        query = "SELECT COUNT(*) FROM run WHERE run_id = ?;"
        if self.query(query, (run_id,))[0]["COUNT(*)"] == 0:
            return set()

        # Get nodes
        query = "SELECT node_id FROM node WHERE online_until > ?;"
        rows = self.query(query, (time.time(),))
        result: Set[int] = {row["node_id"] for row in rows}
        return result

    def get_node_id(self, client_public_key: bytes) -> Optional[int]:
        """Retrieve stored `node_id` filtered by `client_public_keys`."""
        query = "SELECT node_id FROM node WHERE public_key = :public_key;"
        row = self.query(query, {"public_key": client_public_key})
        if len(row) > 0:
            node_id: int = row[0]["node_id"]
            return node_id
        return None

    def create_run(self, fab_id: str, fab_version: str) -> int:
        """Create a new run for the specified `fab_id` and `fab_version`."""
        # Sample a random int64 as run_id
        run_id: int = int.from_bytes(os.urandom(8), "little", signed=True)

        # Check conflicts
        query = "SELECT COUNT(*) FROM run WHERE run_id = ?;"
        # If run_id does not exist
        if self.query(query, (run_id,))[0]["COUNT(*)"] == 0:
            query = "INSERT INTO run (run_id, fab_id, fab_version) VALUES (?, ?, ?);"
            self.query(query, (run_id, fab_id, fab_version))
            return run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve information about the run with the specified `run_id`."""
        query = "SELECT * FROM run WHERE run_id = ?;"
        try:
            row = self.query(query, (run_id,))[0]
            return Run(
                run_id=run_id, fab_id=row["fab_id"], fab_version=row["fab_version"]
            )
        except sqlite3.IntegrityError:
            log(ERROR, "`run_id` does not exist.")
            return None


def dict_factory(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
) -> Dict[str, Any]:
    """Turn SQLite results into dicts.

    Less efficent for retrival of large amounts of data but easier to use.
    """
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))

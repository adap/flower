"""SQLite based implemenation of server state."""
import sqlite3
from typing import List, Optional, Set, Tuple
from uuid import UUID
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns, TaskRes
from flwr.proto.transport_pb2 import ServerMessage, ClientMessage
from .state import State
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

SQL_CREATE_TABLE_TASK_INS = """
CREATE TABLE IF NOT EXISTS task_ins(
    task_id                 TEXT,
    group_id                TEXT,
    workload_id             TEXT,
    producer_anonymous      BOOLEAN,
    producer_node_id        INTEGER,
    consumer_anonymous      BOOLEAN,
    consumer_node_id        INTEGER,
    created_at              TEXT,
    delivered_at            TEXT,
    ttl                     TEXT,
    ancestry                TEXT,
    legacy_server_message   BLOB,
    legacy_client_message   BLOB
);
"""

SQL_CREATE_TABLE_TASK_RES = """
CREATE TABLE IF NOT EXISTS task_res(
    task_id                 TEXT,
    group_id                TEXT,
    workload_id             TEXT,
    producer_anonymous      BOOLEAN,
    producer_node_id        INTEGER,
    consumer_anonymous      BOOLEAN,
    consumer_node_id        INTEGER,
    created_at              TEXT,
    delivered_at            TEXT,
    ttl                     TEXT,
    ancestry                TEXT,
    legacy_server_message   BLOB,
    legacy_client_message   BLOB
);
"""

SQL_CREATE_TABLE_NODE = """
CREATE TABLE IF NOT EXISTS node(
    node_id INTEGER
);
"""


class SqliteState(State):
    """SQLite based state implemenation."""

    def __init__(
        self,
        database_path: str = ":memory:",
    ) -> None:
        """Initialize an SqliteState.

        Parameters
        ----------
        database : (path-like object)
            The path to the database file to be opened. Pass ":memory:" to open
            a connectionto a database that is in RAM instead of on disk.
        """
        self.database_path = database_path
        self.conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> List[Tuple[str]]:
        """Create tables if they don't exist yet."""
        self.conn = sqlite3.connect(self.database_path)
        self.conn.row_factory = dict_factory
        cur = self.conn.cursor()

        # Create each table if not exists queries
        cur.execute(SQL_CREATE_TABLE_TASK_INS)
        cur.execute(SQL_CREATE_TABLE_TASK_RES)
        cur.execute(SQL_CREATE_TABLE_NODE)

        res = cur.execute("SELECT name FROM sqlite_schema;")
        return res.fetchall()

    def _query(self, query: str) -> List[Tuple[any]]:
        if self.conn is None:
            raise Exception("State is not initialized.")

        cur = self.conn.cursor()
        res = cur.execute(query)
        return res.fetchall()

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns.

        Usually, the Driver API calls this to schedule instructions.

        Stores the value of the task_ins in the state and, if successful, returns the
        task_id (UUID) of the task_ins. If, for any reason, storing the task_ins fails,
        `None` is returned.

        Constraints
        -----------
        If `task_ins.task.consumer.anonymous` is `True`, then
        `task_ins.task.consumer.node_id` MUST NOT be set (equal 0). Any implemenation
        may just override it with zero instead of validating.

        If `task_ins.task.consumer.anonymous` is `False`, then
        `task_ins.task.consumer.node_id` MUST be set (not 0)
        """
        if self.conn is None:
            raise Exception("State is not initialized.")

        # Create and set task_id
        task_id = uuid4()
        task_ins.task_id = str(task_id)

        # Set created_at
        created_at: datetime = _now()
        task_ins.task.created_at = created_at.isoformat()

        # Set ttl
        ttl: datetime = created_at + timedelta(hours=24)
        task_ins.task.ttl = ttl.isoformat()

        # Store TaskIns
        data = (task_ins_to_dict(task_ins),)
        columns = ", ".join([f":{key}" for key in data[0].keys()])
        query = f"INSERT INTO task_ins VALUES({columns});"

        cur = self.conn.cursor()
        cur.executemany(query, data)

        return task_id

    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get TaskIns optionally filtered by node_id.

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

        If `delivered_at` MUST BE set (not `""`) otherwise the TaskIns MUST not be in
        the result.

        If `limit` is not `None`, return, at most, `limit` number of `task_ins`. If
        `limit` is set, it has to be greater zero.
        """
        if self.conn is None:
            raise Exception("State is not initialized.")

        cur = self.conn.cursor()
        cur.execute("SELECT * FROM task_ins;")
        rows = cur.fetchall()
        result = [dict_to_task_ins(row) for row in rows]
        return result

    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes.

        Usually, the Fleet API calls this for Nodes returning results.

        Stores the TaskRes and, if successful, returns the `task_id` (UUID) of
        the `task_res`. If storing the `task_res` fails, `None` is returned.

        Constraints
        -----------
        If `task_res.task.consumer.anonymous` is `True`, then
        `task_res.task.consumer.node_id` MUST NOT be set (equal 0). Any implemenation
        may just override it with zero instead of validating.

        If `task_res.task.consumer.anonymous` is `False`, then
        `task_res.task.consumer.node_id` MUST be set (not 0)
        """

    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get TaskRes for task_ids.

        Usually, the Driver API calls this for Nodes planning to work on one or more
        TaskIns.

        Retrieves all TaskRes for the given `task_ids` and returns and empty list of
        none could be found.

        Constraints
        -----------
        If `limit` is not `None`, return, at most, `limit` number of TaskRes. The limit
        will only take effect if enough task_ids are in the set AND are currently
        available. If `limit` is set, it has to be greater zero.
        """

    def num_task_ins(self) -> int:
        """Number of task_ins in store.

        This includes delivered but not yet deleted task_ins.
        """

    def num_task_res(self) -> int:
        """Number of task_res in store.

        This includes delivered but not yet deleted task_res.
        """

    def delete_tasks(self, task_ids: Set[UUID]) -> None:
        """Delete all delivered TaskIns/TaskRes pairs."""

    def register_node(self, node_id: int) -> None:
        """Store `node_id` in state."""

    def unregister_node(self, node_id: int) -> None:
        """Remove `node_id` from state."""

    def get_nodes(self) -> Set[int]:
        """Retrieve all currently stored node IDs as a set."""


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def dict_factory(cursor, row):
    """Used to turn SQLite results into dicts.

    Less efficent for retrival of large amounts of data but easier to use.
    """
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def task_ins_to_dict(task_ins: TaskIns):
    result = {
        "task_id": task_ins.task_id,
        "group_id": task_ins.group_id,
        "workload_id": task_ins.workload_id,
        "producer_anonymous": task_ins.task.producer.anonymous,
        "producer_node_id": task_ins.task.producer.node_id,
        "consumer_anonymous": task_ins.task.consumer.anonymous,
        "consumer_node_id": task_ins.task.consumer.node_id,
        "created_at": task_ins.task.created_at,
        "delivered_at": task_ins.task.delivered_at,
        "ttl": task_ins.task.ttl,
        "ancestry": ",".join(task_ins.task.ancestry),
        "legacy_server_message": task_ins.task.legacy_client_message.SerializeToString(),
        "legacy_client_message": task_ins.task.legacy_server_message.SerializeToString(),
    }
    return result


def dict_to_task_ins(task_ins_dict):
    """Turn task_ins_dict into protobuf message."""
    server_message = ServerMessage()
    server_message.ParseFromString(
        task_ins_dict["legacy_server_message"]
    )
    client_message = ClientMessage()
    client_message.ParseFromString(
        task_ins_dict["legacy_client_message"]
    )

    result = TaskIns(
        task_id=task_ins_dict["task_id"],
        group_id=task_ins_dict["group_id"],
        workload_id=task_ins_dict["workload_id"],
        task=Task(
            producer=Node(
                node_id=task_ins_dict["producer_node_id"],
                anonymous=task_ins_dict["producer_anonymous"],
            ),
            consumer=Node(
                node_id=task_ins_dict["consumer_node_id"],
                anonymous=task_ins_dict["consumer_anonymous"],
            ),
            created_at=task_ins_dict["created_at"],
            delivered_at=task_ins_dict["delivered_at"],
            ttl=task_ins_dict["ttl"],
            ancestry=task_ins_dict["ancestry"].split(","),
            legacy_server_message=server_message,
            legacy_client_message=client_message,
        ),
    )
    return result

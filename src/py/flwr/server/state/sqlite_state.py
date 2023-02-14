import sqlite3
from logging import INFO
from typing import List, Optional, Set
from uuid import UUID

from flwr.common.logger import log
from flwr.proto.task_pb2 import TaskIns, TaskRes

from .state import State

DB_TABLE_NAME = "task"


SQL_CREATE_TABLE = """
CREATE TABLE task(
    task_id primary key,
    group_id,
    workload_id,
    producer_anonymous,
    producer_node_id,
    consumer_anonymous,
    consumer_node_id,
    created_at,
    delivered_at,
    ttl,
    ancestry,
    legacy_server_message,
    legacy_client_message,                        
)
"""


class SqliteState(State):
    def __init__(self, database_path: str = ":memory:") -> None:
        """Initialize an SqliteState.
        
        Parameters
        ----------
        database : (path-like object)
            The path to the database file to be opened. Pass ":memory:" to open
            a connectionto a database that is in RAM instead of on disk.
        """
        self.database_path = database_path

    def init_state(self) -> None:
        """."""
        con = sqlite3.connect(self.database_path)
        cur = con.cursor()

        # Check if the DB has already been initialized
        res = cur.execute(f"SELECT {DB_TABLE_NAME} FROM sqlite_master")
        found = res.fetchone()
        if len(found) == 1 and found[0] == DB_TABLE_NAME:
            log(INFO, "SqliteState: use existing DB")
            return

        # Set up the DB
        cur.execute(SQL_CREATE_TABLE)

    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""

    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""

    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes."""

    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get all TaskRes that have not been delivered yet."""

    def delete_tasks(self, task_ids: Set[UUID]) -> None:
        """Delete all delivered TaskIns/TaskRes pairs."""

    def register_node(self, node_id: int) -> None:
        """Register a client node."""

    def unregister_node(self, node_id: int) -> None:
        """Unregister a client node."""

    def get_nodes(self) -> Set[int]:
        """Return all available client nodes."""

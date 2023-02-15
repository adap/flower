"""SQLite based implemenation of server state."""
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
    """."""

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

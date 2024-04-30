# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Abstract base class State."""


import abc
from typing import List, Optional, Set, Tuple
from uuid import UUID

from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611


class State(abc.ABC):  # pylint: disable=R0904
    """Abstract State."""

    @abc.abstractmethod
    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns.

        Usually, the Driver API calls this to schedule instructions.

        Stores the value of the `task_ins` in the state and, if successful, returns the
        `task_id` (UUID) of the `task_ins`. If, for any reason,
        storing the `task_ins` fails, `None` is returned.

        Constraints
        -----------
        If `task_ins.task.consumer.anonymous` is `True`, then
        `task_ins.task.consumer.node_id` MUST NOT be set (equal 0).

        If `task_ins.task.consumer.anonymous` is `False`, then
        `task_ins.task.consumer.node_id` MUST be set (not 0)

        If `task_ins.run_id` is invalid, then
        storing the `task_ins` MUST fail.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes.

        Usually, the Fleet API calls this for Nodes returning results.

        Stores the TaskRes and, if successful, returns the `task_id` (UUID) of
        the `task_res`. If storing the `task_res` fails, `None` is returned.

        Constraints
        -----------
        If `task_res.task.consumer.anonymous` is `True`, then
        `task_res.task.consumer.node_id` MUST NOT be set (equal 0).

        If `task_res.task.consumer.anonymous` is `False`, then
        `task_res.task.consumer.node_id` MUST be set (not 0)

        If `task_res.run_id` is invalid, then
        storing the `task_res` MUST fail.
        """

    @abc.abstractmethod
    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get TaskRes for task_ids.

        Usually, the Driver API calls this method to get results for instructions it has
        previously scheduled.

        Retrieves all TaskRes for the given `task_ids` and returns and empty list of
        none could be found.

        Constraints
        -----------
        If `limit` is not `None`, return, at most, `limit` number of TaskRes. The limit
        will only take effect if enough task_ids are in the set AND are currently
        available. If `limit` is set, it has to be greater zero.
        """

    @abc.abstractmethod
    def num_task_ins(self) -> int:
        """Calculate the number of task_ins in store.

        This includes delivered but not yet deleted task_ins.
        """

    @abc.abstractmethod
    def num_task_res(self) -> int:
        """Calculate the number of task_res in store.

        This includes delivered but not yet deleted task_res.
        """

    @abc.abstractmethod
    def delete_tasks(self, task_ids: Set[UUID]) -> None:
        """Delete all delivered TaskIns/TaskRes pairs."""

    @abc.abstractmethod
    def create_node(self, ping_interval: float) -> int:
        """Create, store in state, and return `node_id`."""

    @abc.abstractmethod
    def restore_node(self, node_id: int, ping_interval: float) -> bool:
        """Restore `node_id` and return True if succeed."""

    @abc.abstractmethod
    def delete_node(self, node_id: int) -> None:
        """Remove `node_id` from state."""

    @abc.abstractmethod
    def get_nodes(self, run_id: int) -> Set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """

    @abc.abstractmethod
    def get_node_id(self, client_public_key: bytes) -> int:
        """Retrieve stored `node_id` filtered by `client_public_keys`."""

    @abc.abstractmethod
    def store_node_id_and_public_key(self, node_id: int, public_key: bytes) -> None:
        """Store `node_id` and the corresponding `public_key`."""

    @abc.abstractmethod
    def create_run(self, fab_id: str, fab_version: str) -> int:
        """Create a new run for the specified `fab_id` and `fab_version`."""

    @abc.abstractmethod
    def get_run(self, run_id: int) -> Tuple[int, str, str]:
        """Retrieve information about the run with the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run.

        Returns
        -------
        Tuple[int, str, str]
            A tuple containing three elements:
            - `run_id`: The identifier of the run, same as the specified `run_id`.
            - `fab_id`: The identifier of the FAB used in the specified run.
            - `fab_version`: The version of the FAB used in the specified run.
        """

    @abc.abstractmethod
    def store_server_private_public_key(
        self, private_key: bytes, public_key: bytes
    ) -> None:
        """Store `server_private_key` and `server_public_key` in state."""

    @abc.abstractmethod
    def get_server_private_key(self) -> Optional[bytes]:
        """Retrieve `server_private_key` in urlsafe bytes."""

    @abc.abstractmethod
    def get_server_public_key(self) -> Optional[bytes]:
        """Retrieve `server_public_key` in urlsafe bytes."""

    @abc.abstractmethod
    def store_client_public_keys(self, public_keys: Set[bytes]) -> None:
        """Store a set of `client_public_keys` in state."""

    @abc.abstractmethod
    def store_client_public_key(self, public_key: bytes) -> None:
        """Store a `client_public_key` in state."""

    @abc.abstractmethod
    def get_client_public_keys(self) -> Set[bytes]:
        """Retrieve all currently stored `client_public_keys` as a set."""

    @abc.abstractmethod
    def acknowledge_ping(self, node_id: int, ping_interval: float) -> bool:
        """Acknowledge a ping received from a node, serving as a heartbeat.

        Parameters
        ----------
        node_id : int
            The `node_id` from which the ping was received.
        ping_interval : float
            The interval (in seconds) from the current timestamp within which the next
            ping from this node must be received. This acts as a hard deadline to ensure
            an accurate assessment of the node's availability.

        Returns
        -------
        is_acknowledged : bool
            True if the ping is successfully acknowledged; otherwise, False.
        """

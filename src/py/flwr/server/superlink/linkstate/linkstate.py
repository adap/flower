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
"""Abstract base class LinkState."""


import abc
from typing import Optional
from uuid import UUID

from flwr.common import Context
from flwr.common.record import ConfigsRecord
from flwr.common.typing import Run, RunStatus, UserConfig
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611


class LinkState(abc.ABC):  # pylint: disable=R0904
    """Abstract LinkState."""

    @abc.abstractmethod
    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns.

        Usually, the ServerAppIo API calls this to schedule instructions.

        Stores the value of the `task_ins` in the link state and, if successful,
        returns the `task_id` (UUID) of the `task_ins`. If, for any reason,
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
    ) -> list[TaskIns]:
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
    def get_task_res(self, task_ids: set[UUID]) -> list[TaskRes]:
        """Get TaskRes for the given TaskIns IDs.

        This method is typically called by the ServerAppIo API to obtain
        results (TaskRes) for previously scheduled instructions (TaskIns).
        For each task_id provided, this method returns one of the following responses:

        - An error TaskRes if the corresponding TaskIns does not exist or has expired.
        - An error TaskRes if the corresponding TaskRes exists but has expired.
        - The valid TaskRes if the TaskIns has a corresponding valid TaskRes.
        - Nothing if the TaskIns is still valid and waiting for a TaskRes.

        Parameters
        ----------
        task_ids : set[UUID]
            A set of TaskIns IDs for which to retrieve results (TaskRes).

        Returns
        -------
        list[TaskRes]
            A list of TaskRes corresponding to the given task IDs. If no
            TaskRes could be found for any of the task IDs, an empty list is returned.
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
    def delete_tasks(self, task_ins_ids: set[UUID]) -> None:
        """Delete TaskIns/TaskRes pairs based on provided TaskIns IDs.

        Parameters
        ----------
        task_ins_ids : set[UUID]
            A set of TaskIns IDs. For each ID in the set, the corresponding
            TaskIns and its associated TaskRes will be deleted.
        """

    @abc.abstractmethod
    def get_task_ids_from_run_id(self, run_id: int) -> set[UUID]:
        """Get all TaskIns IDs for the given run_id."""

    @abc.abstractmethod
    def create_node(
        self, ping_interval: float, public_key: Optional[bytes] = None
    ) -> int:
        """Create, store in the link state, and return `node_id`."""

    @abc.abstractmethod
    def delete_node(self, node_id: int, public_key: Optional[bytes] = None) -> None:
        """Remove `node_id` from the link state."""

    @abc.abstractmethod
    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """

    @abc.abstractmethod
    def get_node_id(self, node_public_key: bytes) -> Optional[int]:
        """Retrieve stored `node_id` filtered by `node_public_keys`."""

    @abc.abstractmethod
    def create_run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        fab_id: Optional[str],
        fab_version: Optional[str],
        fab_hash: Optional[str],
        override_config: UserConfig,
        federation_options: ConfigsRecord,
    ) -> int:
        """Create a new run for the specified `fab_hash`."""

    @abc.abstractmethod
    def get_run_ids(self) -> set[int]:
        """Retrieve all run IDs."""

    @abc.abstractmethod
    def get_run(self, run_id: int) -> Optional[Run]:
        """Retrieve information about the run with the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run.

        Returns
        -------
        Optional[Run]
            The `Run` instance if found; otherwise, `None`.
        """

    @abc.abstractmethod
    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs.

        Parameters
        ----------
        run_ids : set[int]
            A set of run identifiers for which to retrieve statuses.

        Returns
        -------
        dict[int, RunStatus]
            A dictionary mapping each valid run ID to its corresponding status.

        Notes
        -----
        Only valid run IDs that exist in the State will be included in the returned
        dictionary. If a run ID is not found, it will be omitted from the result.
        """

    @abc.abstractmethod
    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run.
        new_status : RunStatus
            The new status to be assigned to the run.

        Returns
        -------
        bool
            True if the status update is successful; False otherwise.
        """

    @abc.abstractmethod
    def get_pending_run_id(self) -> Optional[int]:
        """Get the `run_id` of a run with `Status.PENDING` status.

        Returns
        -------
        Optional[int]
            The `run_id` of a `Run` that is pending to be started; None if
            there is no Run pending.
        """

    @abc.abstractmethod
    def get_federation_options(self, run_id: int) -> Optional[ConfigsRecord]:
        """Retrieve the federation options for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run.

        Returns
        -------
        Optional[ConfigsRecord]
            The federation options for the run if it exists; None otherwise.
        """

    @abc.abstractmethod
    def store_server_private_public_key(
        self, private_key: bytes, public_key: bytes
    ) -> None:
        """Store `server_private_key` and `server_public_key` in the link state."""

    @abc.abstractmethod
    def get_server_private_key(self) -> Optional[bytes]:
        """Retrieve `server_private_key` in urlsafe bytes."""

    @abc.abstractmethod
    def get_server_public_key(self) -> Optional[bytes]:
        """Retrieve `server_public_key` in urlsafe bytes."""

    @abc.abstractmethod
    def clear_supernode_auth_keys_and_credentials(self) -> None:
        """Clear stored `node_public_keys` and credentials in the link state if any."""

    @abc.abstractmethod
    def store_node_public_keys(self, public_keys: set[bytes]) -> None:
        """Store a set of `node_public_keys` in the link state."""

    @abc.abstractmethod
    def store_node_public_key(self, public_key: bytes) -> None:
        """Store a `node_public_key` in the link state."""

    @abc.abstractmethod
    def get_node_public_keys(self) -> set[bytes]:
        """Retrieve all currently stored `node_public_keys` as a set."""

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

    @abc.abstractmethod
    def get_serverapp_context(self, run_id: int) -> Optional[Context]:
        """Get the context for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to retrieve the context.

        Returns
        -------
        Optional[Context]
            The context associated with the specified `run_id`, or `None` if no context
            exists for the given `run_id`.
        """

    @abc.abstractmethod
    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to set the context.
        context : Context
            The context to be associated with the specified `run_id`.
        """

    @abc.abstractmethod
    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to add a log entry.
        log_message : str
            The log entry to be added to the ServerApp logs.
        """

    @abc.abstractmethod
    def get_serverapp_log(
        self, run_id: int, after_timestamp: Optional[float]
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to retrieve the ServerApp logs.

        after_timestamp : Optional[float]
            Retrieve logs after this timestamp. If set to `None`, retrieve all logs.

        Returns
        -------
        tuple[str, float]
            A tuple containing:
            - The ServerApp logs associated with the specified `run_id`.
            - The timestamp of the latest log entry in the returned logs.
              Returns `0` if no logs are returned.
        """

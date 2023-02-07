# Copyright 2022 Adap GmbH. All Rights Reserved.
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
from typing import List, Optional, Set
from uuid import UUID

from flwr.proto.task_pb2 import TaskIns, TaskRes


class State(abc.ABC):
    """Abstract State."""

    @abc.abstractmethod
    def store_task_ins(self, task_ins: TaskIns) -> Optional[UUID]:
        """Store one TaskIns."""

    @abc.abstractmethod
    def get_task_ins(
        self, node_id: Optional[int], limit: Optional[int]
    ) -> List[TaskIns]:
        """Get all TaskIns that have not been delivered yet."""

    @abc.abstractmethod
    def store_task_res(self, task_res: TaskRes) -> Optional[UUID]:
        """Store one TaskRes."""

    @abc.abstractmethod
    def get_task_res(self, task_ids: Set[UUID], limit: Optional[int]) -> List[TaskRes]:
        """Get all TaskRes that have not been delivered yet."""

    @abc.abstractmethod
    def register_node(self, node_id: int) -> None:
        """Register a client node."""

    @abc.abstractmethod
    def unregister_node(self, node_id: int) -> None:
        """Unregister a client node."""

    @abc.abstractmethod
    def get_nodes(self) -> Set[int]:
        """Return all available client nodes."""

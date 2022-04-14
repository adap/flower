# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Flower TaskManager."""


import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import INFO
from typing import Dict, List, Optional
from xmlrpc.client import Server

from flwr.common.logger import log
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


@dataclass
class Task:
    server_message: ServerMessage
    client_message: Optional[ClientMessage] = None


class TaskManager(ABC):
    """Abstract base class for managing tasks and results."""

    @abstractmethod
    def set_task(self, cid: str, server_message: ServerMessage) -> bool:
        """."""

    @abstractmethod
    def get_task(self, cid: str) -> ServerMessage:
        """."""

    @abstractmethod
    def set_result(self, cid: str, client_message: ClientMessage) -> bool:
        """."""

    @abstractmethod
    def get_result(self, cid: str) -> ClientMessage:
        """."""


class SimpleTaskManager(TaskManager):
    """Manages client tasks and task results."""

    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self._cv = threading.Condition()

    def set_task(self, cid: str, server_message: ServerMessage) -> bool:
        """."""
        with self._cv:
            if cid in self.tasks:
                # raise Exception(f"Task for {cid} already set")
                return False
            self.tasks[cid] = Task(server_message=server_message)
            return True

    def get_task(self, cid: str) -> Optional[ServerMessage]:
        """."""
        with self._cv:
            if cid not in self.tasks:
                return None
            task = self.tasks[cid]
            return task.server_message

    def set_result(self, cid: str, client_message: ClientMessage) -> bool:
        """."""
        with self._cv:
            if cid not in self.tasks:
                return False
            if self.tasks[cid].client_message is not None:
                raise (Exception("Task result already set"))
            task = self.tasks[cid]
            task.client_message = client_message
            self._cv.notify_all()

    def get_result(self, cid: str) -> ClientMessage:
        """."""
        with self._cv:
            if cid not in self.tasks:
                raise Exception(f"Requesting result for unknown {cid}")
            self._cv.wait_for(
                predicate=lambda: cid in self.tasks
                and self.tasks[cid].client_message is not None
            )
            task = self.tasks[cid]
            del self.tasks[cid]
            return task.client_message

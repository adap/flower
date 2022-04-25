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
"""Global state, this should probably be replaced by a different mechanism."""


from distutils.log import debug
from logging import ERROR

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.task_manager import TaskManager


class State(object):
    """The State singleton is a container to hold shared references.

    There are to several components that are needed to facilitate communication
    between the main thread (running the Flower server's fit method) and the
    REST API server thread (running the FastAPI server through uvicorn).

    At the time of writing, there are two shared components:
    - ClientManager, manages the set of currently available clients
    - TaskManager, manages client tasks and associated task results
    """

    _instance = None

    _client_manager = None
    _task_manager = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        # FIXME this is not threadsafe
        if cls._instance is None:
            print("Creating new instance")
            cls._instance = cls.__new__(cls)
        return cls._instance

    def set_client_manager(self, client_manager: ClientManager) -> None:
        self._client_manager = client_manager

    def get_client_manager(self) -> ClientManager:
        return self._client_manager

    def set_task_manager(self, task_manager: TaskManager) -> None:
        self._task_manager = task_manager

    def get_task_manager(self) -> TaskManager:
        return self._task_manager

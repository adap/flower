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
"""Flower app."""


from dataclasses import dataclass
from typing import Callable, Dict

from flwr.client.message_handler.message_handler import handle
from flwr.client.typing import ClientFn
from flwr.proto.task_pb2 import TaskIns, TaskRes
from flwr.client.workload_state import WorkloadState

@dataclass
class Fwd:
    """."""

    task_ins: TaskIns
    state: WorkloadState


@dataclass
class Bwd:
    """."""

    task_res: TaskRes
    state: WorkloadState


App = Callable[[Fwd], Bwd]


class Flower:
    """Flower app class."""

    def __init__(
        self,
        client_fn: ClientFn,  # Only for backward compatibility
    ) -> None:
        self.client_fn = client_fn

    def __call__(self, fwd: Fwd) -> Bwd:
        """."""
        # Execute the task
        task_res = handle(
            client_fn=self.client_fn,
            task_ins=fwd.task_ins,
        )
        return Bwd(
            task_res=task_res,
            state=WorkloadState(state={}),
        )

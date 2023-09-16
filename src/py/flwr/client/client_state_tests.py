# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# \
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Client state tests."""

import uuid
from typing import Dict, Optional

from flwr.client import ClientFn, ClientState, WorkloadState
from flwr.common import Config, Scalar
from flwr.proto.node_pb2 import Node
from flwr.proto.task_pb2 import Task, TaskIns
from flwr.proto.transport_pb2 import ServerMessage

from .message_handler.message_handler import handle
from .numpy_client import NumPyClient


class SimpleClientWithStateInteraction(NumPyClient):
    """A simple client that interacts with its state."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Call parent get_properties method and interact with the client state."""
        if hasattr(self.state, "counter"):
            # Increment counter
            self.state.counter += 1  # type:ignore
        else:
            # Add a new attribute to the state
            self.state.counter = 1  # type:ignore

        return super().get_properties(config)


def prepare_simple_get_properties_task_ins(workload_id: str) -> TaskIns:
    """Return get properties task instructions."""
    ins = ServerMessage.GetPropertiesIns()
    task_ins = TaskIns(
        task_id=str(uuid.uuid4()),
        group_id="",
        workload_id=workload_id,
        task=Task(
            producer=Node(node_id=0, anonymous=True),
            consumer=Node(node_id=0, anonymous=True),
            ancestry=[],
            legacy_server_message=ServerMessage(get_properties_ins=ins),
        ),
    )
    return task_ins


def get_client_fn() -> ClientFn:
    """Return callable for client generation."""

    def client_fn(
        cid: str,  # pylint: disable=unused-argument
    ) -> SimpleClientWithStateInteraction:
        return SimpleClientWithStateInteraction()

    return client_fn


def basic_task_execution_flow(
    task_ins: TaskIns, client_state: ClientState
) -> Optional[WorkloadState]:  # pragma: no cover
    """Run standard task-execution pipeline."""
    # Prepare state
    client_state.register_workload(task_ins.workload_id)  # pylint: disable=no-member
    workload_state = client_state[task_ins.workload_id]  # pylint: disable=no-member

    # Execute
    _, _, _, workload_state_updated = handle(
        client_fn=get_client_fn(),
        task_ins=task_ins,
        workload_state=workload_state,
    )
    return workload_state_updated


def test_basic_stateful_client_workflow() -> None:
    """Create one client that interacts with its state at a basic level.

    This tests primarily that the clients: (1) get their state injected correctly;
    and (2) that their state is indeed preserved across client re-instantiation events.
    """
    client_state = ClientState()

    tasks = [
        prepare_simple_get_properties_task_ins("A"),
        prepare_simple_get_properties_task_ins("A"),
        prepare_simple_get_properties_task_ins("B"),
        prepare_simple_get_properties_task_ins("B"),
        prepare_simple_get_properties_task_ins("A"),
    ]

    def run_one_task(task_ins: TaskIns) -> None:
        # Run task
        workload_state = basic_task_execution_flow(task_ins, client_state)

        # Update state
        client_state.update_workload_state(workload_state)

    for i, task in enumerate(tasks):
        run_one_task(task)

        if i < 2:
            # Running task A
            assert client_state["A"].counter == i + 1  # type:ignore
            assert len(client_state.workload_states) == 1
        if 1 < i < 4:
            # Now running task B
            assert client_state["A"].counter == 2  # type:ignore
            assert len(client_state.workload_states) == 2
            assert client_state["B"].counter == i - 1  # type:ignore
        if i == 4:
            # Running task A one final time
            assert client_state["A"].counter == 3  # type:ignore
            assert client_state["B"].counter == 2  # type:ignore
            assert len(client_state.workload_states) == 2

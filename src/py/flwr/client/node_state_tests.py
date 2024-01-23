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
"""Node state tests."""


from flwr.client.node_state import NodeState
from flwr.client.run_state import RunState
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611


def _run_dummy_task(state: RunState) -> RunState:
    if "counter" in state.state:
        state.state["counter"] += "1"
    else:
        state.state["counter"] = "1"

    return state


def test_multirun_in_node_state() -> None:
    """Test basic NodeState logic."""
    # Tasks to perform
    tasks = [TaskIns(run_id=run_id) for run_id in [0, 1, 1, 2, 3, 2, 1, 5]]
    # the "tasks" is to count how many times each run is executed
    expected_values = {0: "1", 1: "1" * 3, 2: "1" * 2, 3: "1", 5: "1"}

    # NodeState
    node_state = NodeState()

    for task in tasks:
        run_id = task.run_id

        # Register
        node_state.register_runstate(run_id=run_id)

        # Get run state
        state = node_state.retrieve_runstate(run_id=run_id)

        # Run "task"
        updated_state = _run_dummy_task(state)

        # Update run state
        node_state.update_runstate(run_id=run_id, run_state=updated_state)

    # Verify values
    for run_id, state in node_state.run_states.items():
        assert state.state["counter"] == expected_values[run_id]

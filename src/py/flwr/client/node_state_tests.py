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


from typing import cast

from flwr.client.node_state import NodeState
from flwr.common import ConfigsRecord, Context
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611


def _run_dummy_task(context: Context) -> Context:
    counter_value: str = "1"
    if "counter" in context.state.configs_records.keys():
        counter_value = cast(str, context.state.configs_records["counter"]["count"])
        counter_value += "1"

    context.state.configs_records["counter"] = ConfigsRecord({"count": counter_value})

    return context


def test_multirun_in_node_state() -> None:
    """Test basic NodeState logic."""
    # Tasks to perform
    tasks = [TaskIns(run_id=run_id) for run_id in [0, 1, 1, 2, 3, 2, 1, 5]]
    # the "tasks" is to count how many times each run is executed
    expected_values = {0: "1", 1: "1" * 3, 2: "1" * 2, 3: "1", 5: "1"}

    # NodeState
    node_state = NodeState(partition_id=None)

    for task in tasks:
        run_id = task.run_id

        # Register
        node_state.register_context(run_id=run_id)

        # Get run state
        context = node_state.retrieve_context(run_id=run_id)

        # Run "task"
        updated_state = _run_dummy_task(context)

        # Update run state
        node_state.update_context(run_id=run_id, context=updated_state)

    # Verify values
    for run_id, run_info in node_state.run_info_dict.items():
        assert (
            run_info.context.state.configs_records["counter"]["count"]
            == expected_values[run_id]
        )

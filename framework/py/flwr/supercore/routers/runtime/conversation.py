# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Load raw AgentApp conversation events from a RunSeries."""

from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.typing import ConversationSummary

_CONVERSATION_EVENT_TYPES = frozenset(
    {"message", "response.completed", "function_call", "function_call_output"}
)


def load_conversation(
    state: CoreState, series_id: int
) -> tuple[list[TaskEvent], ConversationSummary | None]:
    """Return AgentApp events after the RunSeries summary cursor, and the summary."""
    # Find the runs belonging to this conversation.
    series = state.get_run_series(series_ids=[series_id])
    if not series:
        raise ValueError(f"RunSeries {series_id} not found")

    # Load the summary and identify the AgentApp tasks that hold conversation events.
    summary = state.get_run_series_summary(series_id)
    run_ids = series[0].run_ids
    agent_task_ids = [
        task.task_id
        for task in state.get_tasks(run_ids=run_ids)
        if task.type == TaskType.AGENT_APP
    ]

    # Read only events newer than the summary, then keep conversation event types.
    events = state.get_task_events(
        run_ids=run_ids,
        task_ids=agent_task_ids,
        after_task_event_id=summary.through_event_id if summary else None,
    )
    return [
        event for event in events if event.event in _CONVERSATION_EVENT_TYPES
    ], summary

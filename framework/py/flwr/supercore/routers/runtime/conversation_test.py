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
"""Tests for loading RunSeries conversation events."""

from unittest.mock import Mock, create_autospec

import pytest

from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.typing import ConversationSummary

from .conversation import load_conversation


def _state() -> Mock:
    state: Mock = create_autospec(CoreState, instance=True)
    state.get_run_series.return_value = [RunSeries(series_id=42, run_ids=[10, 11])]
    state.get_tasks.return_value = [
        Task(task_id=100, run_id=10, type=TaskType.AGENT_APP),
        Task(task_id=101, run_id=10, type=TaskType.MODEL),
        Task(task_id=200, run_id=11, type=TaskType.AGENT_APP),
    ]
    return state


def test_load_conversation_after_summary_cursor() -> None:
    """Load only conversation events from AgentApp tasks after the cursor."""
    state = _state()
    summary = ConversationSummary("Earlier work", through_event_id=4)
    state.get_run_series_summary.return_value = summary
    state.get_task_events.return_value = [
        TaskEvent(id=5, event="message"),
        TaskEvent(id=6, event="response.output_text.delta"),
        TaskEvent(id=7, event="function_call"),
        TaskEvent(id=8, event="function_call_output"),
        TaskEvent(id=9, event="response.completed"),
    ]

    events, loaded_summary = load_conversation(state, 42)

    assert [event.id for event in events] == [5, 7, 8, 9]
    assert loaded_summary == summary
    state.get_task_events.assert_called_once_with(
        run_ids=[10, 11], task_ids=[100, 200], after_task_event_id=4
    )


def test_load_conversation_without_summary() -> None:
    """Read from the start when the RunSeries has no summary."""
    state = _state()
    state.get_run_series_summary.return_value = None
    state.get_task_events.return_value = [TaskEvent(id=1, event="message")]

    events, summary = load_conversation(state, 42)

    assert [event.id for event in events] == [1]
    assert summary is None
    state.get_task_events.assert_called_once_with(
        run_ids=[10, 11], task_ids=[100, 200], after_task_event_id=None
    )


def test_load_conversation_rejects_unknown_series() -> None:
    """Do not query events for a nonexistent RunSeries."""
    state = _state()
    state.get_run_series.return_value = []

    with pytest.raises(ValueError, match="RunSeries 42 not found"):
        load_conversation(state, 42)

    state.get_task_events.assert_not_called()

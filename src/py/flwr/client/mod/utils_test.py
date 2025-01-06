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
"""Tests for the utility functions."""


import unittest
from typing import cast

from flwr.client.typing import ClientAppCallable, Mod
from flwr.common import (
    DEFAULT_TTL,
    ConfigsRecord,
    Context,
    Message,
    Metadata,
    MetricsRecord,
    RecordSet,
)

from .utils import make_ffn

METRIC = "context"
COUNTER = "counter"


def _increment_context_counter(context: Context) -> None:
    # Read from context
    current_counter = cast(int, context.state.metrics_records[METRIC][COUNTER])
    # update and override context
    current_counter += 1
    context.state.metrics_records[METRIC] = MetricsRecord({COUNTER: current_counter})


def make_mock_mod(name: str, footprint: list[str]) -> Mod:
    """Make a mock mod."""

    def mod(message: Message, context: Context, app: ClientAppCallable) -> Message:
        footprint.append(name)
        # add empty ConfigRecord to in_message for this mod
        message.content.configs_records[name] = ConfigsRecord()
        _increment_context_counter(context)
        out_message: Message = app(message, context)
        footprint.append(name)
        _increment_context_counter(context)
        # add empty ConfigRegcord to out_message for this mod
        out_message.content.configs_records[name] = ConfigsRecord()
        return out_message

    return mod


def make_mock_app(name: str, footprint: list[str]) -> ClientAppCallable:
    """Make a mock app."""

    def app(message: Message, context: Context) -> Message:
        footprint.append(name)
        message.content.configs_records[name] = ConfigsRecord()
        out_message = Message(metadata=message.metadata, content=RecordSet())
        out_message.content.configs_records[name] = ConfigsRecord()
        print(context)
        return out_message

    return app


def _get_dummy_flower_message() -> Message:
    return Message(
        content=RecordSet(),
        metadata=Metadata(
            run_id=0,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message="",
            ttl=DEFAULT_TTL,
            message_type="mock",
        ),
    )


class TestMakeApp(unittest.TestCase):
    """Tests for the `make_app` function."""

    def test_multiple_mods(self) -> None:
        """Test if multiple mods are called in the correct order."""
        # Prepare
        footprint: list[str] = []
        mock_app = make_mock_app("app", footprint)
        mock_mod_names = [f"mod{i}" for i in range(1, 15)]
        mock_mods = [make_mock_mod(name, footprint) for name in mock_mod_names]

        state = RecordSet()
        state.metrics_records[METRIC] = MetricsRecord({COUNTER: 0.0})
        context = Context(
            run_id=1, node_id=0, node_config={}, state=state, run_config={}
        )
        message = _get_dummy_flower_message()

        # Execute
        wrapped_app = make_ffn(mock_app, mock_mods)
        out_message = wrapped_app(message, context)

        # Assert
        trace = mock_mod_names + ["app"]
        self.assertEqual(footprint, trace + list(reversed(mock_mod_names)))
        # pylint: disable-next=no-member
        self.assertEqual(
            "".join(message.content.configs_records.keys()), "".join(trace)
        )
        self.assertEqual(
            "".join(out_message.content.configs_records.keys()),
            "".join(reversed(trace)),
        )
        self.assertEqual(state.metrics_records[METRIC][COUNTER], 2 * len(mock_mods))

    def test_filter(self) -> None:
        """Test if a mod can filter incoming TaskIns."""
        # Prepare
        footprint: list[str] = []
        mock_app = make_mock_app("app", footprint)
        context = Context(
            run_id=1, node_id=0, node_config={}, state=RecordSet(), run_config={}
        )
        message = _get_dummy_flower_message()

        def filter_mod(
            message: Message,
            _1: Context,
            _2: ClientAppCallable,
        ) -> Message:
            footprint.append("filter")
            message.content.configs_records["filter"] = ConfigsRecord()
            out_message = Message(metadata=message.metadata, content=RecordSet())
            out_message.content.configs_records["filter"] = ConfigsRecord()
            # Skip calling app
            return out_message

        # Execute
        wrapped_app = make_ffn(mock_app, [filter_mod])
        out_message = wrapped_app(message, context)

        # Assert
        self.assertEqual(footprint, ["filter"])
        # pylint: disable-next=no-member
        self.assertEqual(list(message.content.configs_records.keys())[0], "filter")
        self.assertEqual(list(out_message.content.configs_records.keys())[0], "filter")

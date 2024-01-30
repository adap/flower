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
"""Tests for the utility functions."""


import unittest
from typing import List

from flwr.client.typing import FlowerCallable, Layer
from flwr.common.configsrecord import ConfigsRecord
from flwr.common.context import Context
from flwr.common.message import Message, Metadata
from flwr.common.metricsrecord import MetricsRecord
from flwr.common.recordset import RecordSet

from .utils import make_ffn

METRIC = "context"
COUNTER = "counter"


def _increment_context_counter(context: Context) -> None:
    # Read from context
    current_counter: int = context.state.get_metrics(METRIC)[COUNTER]  # type: ignore
    # update and override context
    current_counter += 1
    context.state.set_metrics(METRIC, record=MetricsRecord({COUNTER: current_counter}))


def make_mock_middleware(name: str, footprint: List[str]) -> Layer:
    """Make a mock middleware layer."""

    def middleware(message: Message, context: Context, app: FlowerCallable) -> Message:
        footprint.append(name)
        # add empty ConfigRecord to in_message for this middleware layer
        message.message.set_configs(name=name, record=ConfigsRecord())
        _increment_context_counter(context)
        out_message: Message = app(message, context)
        footprint.append(name)
        _increment_context_counter(context)
        # add empty ConfigRegcord to out_message for this middleware layer
        out_message.message.set_configs(name=name, record=ConfigsRecord())
        return out_message

    return middleware


def make_mock_app(name: str, footprint: List[str]) -> FlowerCallable:
    """Make a mock app."""

    def app(message: Message, context: Context) -> Message:
        footprint.append(name)
        message.message.set_configs(name=name, record=ConfigsRecord())
        out_message = Message(metadata=message.metadata, message=RecordSet())
        out_message.message.set_configs(name=name, record=ConfigsRecord())
        print(context)
        return out_message

    return app


def _get_dummy_flower_message() -> Message:
    return Message(
        message=RecordSet(),
        metadata=Metadata(run_id=0, task_id="", group_id="", ttl="", task_type="mock"),
    )


class TestMakeApp(unittest.TestCase):
    """Tests for the `make_app` function."""

    def test_multiple_middlewares(self) -> None:
        """Test if multiple middlewares are called in the correct order."""
        # Prepare
        footprint: List[str] = []
        mock_app = make_mock_app("app", footprint)
        mock_middleware_names = [f"middleware{i}" for i in range(1, 15)]
        mock_middleware_layers = [
            make_mock_middleware(name, footprint) for name in mock_middleware_names
        ]

        state = RecordSet()
        state.set_metrics(METRIC, record=MetricsRecord({COUNTER: 0.0}))
        context = Context(state=state)
        message = _get_dummy_flower_message()

        # Execute
        wrapped_app = make_ffn(mock_app, mock_middleware_layers)
        out_message = wrapped_app(message, context)

        # Assert
        trace = mock_middleware_names + ["app"]
        self.assertEqual(footprint, trace + list(reversed(mock_middleware_names)))
        # pylint: disable-next=no-member
        self.assertEqual("".join(message.message.configs.keys()), "".join(trace))
        self.assertEqual(
            "".join(out_message.message.configs.keys()), "".join(reversed(trace))
        )
        self.assertEqual(
            state.get_metrics(METRIC)[COUNTER], 2 * len(mock_middleware_layers)
        )

    def test_filter(self) -> None:
        """Test if a middleware can filter incoming TaskIns."""
        # Prepare
        footprint: List[str] = []
        mock_app = make_mock_app("app", footprint)
        context = Context(state=RecordSet())
        message = _get_dummy_flower_message()

        def filter_layer(
            message: Message,
            _1: Context,
            _2: FlowerCallable,
        ) -> Message:
            footprint.append("filter")
            message.message.set_configs(name="filter", record=ConfigsRecord())
            out_message = Message(metadata=message.metadata, message=RecordSet())
            out_message.message.set_configs(name="filter", record=ConfigsRecord())
            # Skip calling app
            return out_message

        # Execute
        wrapped_app = make_ffn(mock_app, [filter_layer])
        out_message = wrapped_app(message, context)

        # Assert
        self.assertEqual(footprint, ["filter"])
        # pylint: disable-next=no-member
        self.assertEqual(list(message.message.configs.keys())[0], "filter")
        self.assertEqual(list(out_message.message.configs.keys())[0], "filter")

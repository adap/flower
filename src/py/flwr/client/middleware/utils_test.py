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

from flwr.client.run_state import RunState
from flwr.client.typing import Bwd, FlowerCallable, Fwd, Layer
from flwr.common.configsrecord import ConfigsRecord
from flwr.common.flowercontext import FlowerContext, Metadata
from flwr.common.recordset import RecordSet
from flwr.proto.task_pb2 import TaskIns, TaskRes  # pylint: disable=E0611

from .utils import make_ffn


def make_mock_middleware(name: str, footprint: List[str]) -> Layer:
    """Make a mock middleware layer."""

    def middleware(context: FlowerContext, app: FlowerCallable) -> FlowerContext:
        footprint.append(name)
        context.in_message.set_configs(name=name, record=ConfigsRecord())
        ctx: FlowerContext = app(context)
        footprint.append(name)
        ctx.out_message.set_configs(name=name, record=ConfigsRecord())
        return ctx

    return middleware


def make_mock_app(name: str, footprint: List[str]) -> FlowerCallable:
    """Make a mock app."""

    def app(context: FlowerContext) -> FlowerContext:
        footprint.append(name)
        context.in_message.set_configs(name=name, record=ConfigsRecord())
        context.out_message.set_configs(name=name, record=ConfigsRecord())
        return context

    return app


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

        context = FlowerContext(
            in_message=RecordSet(),
            out_message=RecordSet(),
            local=RecordSet(),
            metadata=Metadata(
                run_id=0, task_id="", group_id="", ttl="", task_type="mock"
            ),
        )

        # Execute
        wrapped_app = make_ffn(mock_app, mock_middleware_layers)
        context_ = wrapped_app(context)

        # Assert
        trace = mock_middleware_names + ["app"]
        self.assertEqual(footprint, trace + list(reversed(mock_middleware_names)))
        # pylint: disable-next=no-member
        self.assertEqual("".join(context_.in_message.configs.keys()), "".join(trace))
        self.assertEqual(
            "".join(context_.out_message.configs.keys()), "".join(reversed(trace))
        )

    def test_filter(self) -> None:
        """Test if a middleware can filter incoming TaskIns."""
        # Prepare
        footprint: List[str] = []
        mock_app = make_mock_app("app", footprint)
        task_ins = TaskIns()

        def filter_layer(fwd: Fwd, _: FlowerCallable) -> Bwd:
            footprint.append("filter")
            fwd.task_ins.task_id += "filter"
            # Skip calling app
            return Bwd(task_res=TaskRes(task_id="filter"), state=RunState({}))

        # Execute
        wrapped_app = make_ffn(mock_app, [filter_layer])
        task_res = wrapped_app(Fwd(task_ins=task_ins, state=RunState({}))).task_res

        # Assert
        self.assertEqual(footprint, ["filter"])
        # pylint: disable-next=no-member
        self.assertEqual(task_ins.task_id, "filter")
        self.assertEqual(task_res.task_id, "filter")

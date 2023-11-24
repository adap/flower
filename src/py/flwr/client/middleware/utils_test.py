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

from flwr.client.flower import Bwd, Fwd
from flwr.client.workload_state import WorkloadState
from flwr.proto.task_pb2 import TaskIns, TaskRes

from .typing import App, Layer
from .utils import make_app


def make_mock_middleware(name: str, footprint: List[str]) -> Layer:
    """Make a mock middleware layer."""

    def middleware(fwd: Fwd, app: App) -> Bwd:
        footprint.append(name)
        fwd.task_ins.task_id += f"{name}"
        bwd = app(fwd)
        footprint.append(name)
        bwd.task_res.task_id += f"{name}"
        return bwd

    return middleware


def make_mock_app(name: str, footprint: List[str]) -> App:
    """Make a mock app."""

    def app(fwd: Fwd) -> Bwd:
        footprint.append(name)
        fwd.task_ins.task_id += f"{name}"
        return Bwd(task_res=TaskRes(task_id=name), state=WorkloadState({}))

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
        task_ins = TaskIns()

        # Execute
        wrapped_app = make_app(mock_app, mock_middleware_layers)
        task_res = wrapped_app(Fwd(task_ins=task_ins, state=WorkloadState({}))).task_res

        # Assert
        trace = mock_middleware_names + ["app"]
        self.assertEqual(footprint, trace + list(reversed(mock_middleware_names)))
        # pylint: disable-next=no-member
        self.assertEqual(task_ins.task_id, "".join(trace))
        self.assertEqual(task_res.task_id, "".join(reversed(trace)))

    def test_filter(self) -> None:
        """Test if a middleware can filter incoming TaskIns."""
        # Prepare
        footprint: List[str] = []
        mock_app = make_mock_app("app", footprint)
        task_ins = TaskIns()

        def filter_layer(fwd: Fwd, _: App) -> Bwd:
            footprint.append("filter")
            fwd.task_ins.task_id += "filter"
            # Skip calling app
            return Bwd(task_res=TaskRes(task_id="filter"), state=WorkloadState({}))

        # Execute
        wrapped_app = make_app(mock_app, [filter_layer])
        task_res = wrapped_app(Fwd(task_ins=task_ins, state=WorkloadState({}))).task_res

        # Assert
        self.assertEqual(footprint, ["filter"])
        # pylint: disable-next=no-member
        self.assertEqual(task_ins.task_id, "filter")
        self.assertEqual(task_res.task_id, "filter")

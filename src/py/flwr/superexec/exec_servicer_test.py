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
"""Test the SuperExec API servicer."""


import subprocess
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock

from flwr.common import ConfigsRecord, now
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    StartRunRequest,
)
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkStateFactory

from .exec_servicer import ExecServicer


def test_start_run() -> None:
    """Test StartRun method of ExecServicer."""
    run_res = MagicMock()
    run_res.run_id = 10
    with subprocess.Popen(
        ["echo", "success"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        run_res.proc = proc

    executor = MagicMock()
    executor.start_run = lambda _, __, ___: run_res.run_id

    context_mock = MagicMock()

    request = StartRunRequest()
    request.fab.content = b"test"

    # Create a instance of FlowerServiceServicer
    servicer = ExecServicer(Mock(), Mock(), executor=executor)

    # Execute
    response = servicer.StartRun(request, context_mock)
    assert response.run_id == 10


class TestExecServicer(unittest.TestCase):
    """Test the Exec API servicer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.servicer = ExecServicer(
            linkstate_factory=LinkStateFactory(":flwr-in-memory-state:"),
            ffs_factory=FfsFactory("./tmp"),
            executor=Mock(),
        )
        self.state = self.servicer.linkstate_factory.state()

    def test_list_runs(self) -> None:
        """Test List method of ExecServicer with --runs option."""
        # Prepare
        run_ids = set()
        for _ in range(3):
            run_id = self.state.create_run(
                "mock fabid", "mock fabver", "fake hash", {}, ConfigsRecord()
            )
            run_ids.add(run_id)

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(set(response.run_dict.keys()), run_ids)

    def test_list_run_id(self) -> None:
        """Test List method of ExecServicer with --run-id option."""
        # Prepare
        for _ in range(3):
            run_id = self.state.create_run(
                "mock fabid", "mock fabver", "fake hash", {}, ConfigsRecord()
            )

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(run_id=run_id), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertLess(abs(retrieved_timestamp - now().timestamp()), 1e-3)
        self.assertEqual(set(response.run_dict.keys()), {run_id})

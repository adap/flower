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
from unittest.mock import MagicMock

from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611

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
    executor.start_run = lambda _, __: run_res

    context_mock = MagicMock()

    request = StartRunRequest()
    request.fab_file = b"test"

    # Create a instance of FlowerServiceServicer
    servicer = ExecServicer(executor=executor)

    # Execute
    response = servicer.StartRun(request, context_mock)

    assert response.run_id == 10

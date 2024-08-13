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
"""Test for Flower command line interface `log` command."""


import unittest
from unittest.mock import Mock, patch

from flwr.proto.exec_pb2 import StreamLogsResponse  # pylint: disable=E0611

from .log import print_logs, stream_logs


class TestFlwrLog(unittest.TestCase):
    """Unit tests for `flwr log` CLI functions."""

    def setUp(self) -> None:
        """Initialize mock ExecStub before each test."""
        mock_response_iterator = iter(
            [StreamLogsResponse(log_output=f"result_{i}") for i in range(1, 4)]
        )
        self.mock_stub = Mock()
        self.mock_stub.StreamLogs.return_value = mock_response_iterator
        self.patcher = patch("flwr.cli.log.ExecStub", return_value=self.mock_stub)
        self.patcher.start()

        # Create mock channel
        self.mock_channel = Mock()

    def tearDown(self) -> None:
        """Cleanup."""
        self.patcher.stop()

    def test_flwr_log_stream_method(self) -> None:
        """Test stream_logs."""
        with patch("builtins.print") as mock_print:
            stream_logs(run_id=123, channel=self.mock_channel, duration=1)
            mock_print.assert_any_call("result_1")
            mock_print.assert_any_call("result_2")
            mock_print.assert_any_call("result_3")

    def test_flwr_log_print_method(self) -> None:
        """Test print_logs."""
        with patch("builtins.print") as mock_print:
            print_logs(run_id=123, channel=self.mock_channel, timeout=0, is_test=True)
            mock_print.assert_any_call("result_1")
            mock_print.assert_any_call("result_2")
            mock_print.assert_any_call("result_3")

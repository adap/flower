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

    @patch("flwr.cli.log.ExecStub")
    def test_flwr_log_stream_method(self, mock_stub: Mock) -> None:
        """Test stream_logs."""
        # Create mock response iterator
        mock_response_iterator = iter(
            [StreamLogsResponse(log_output=f"print_result_{i}") for i in range(1, 4)]
        )

        # Set up stub
        mock_stub_instance = mock_stub.return_value
        mock_stub_instance.StreamLogs.return_value = mock_response_iterator

        # Create mock channel
        mock_channel = Mock()

        with patch("builtins.print") as mock_print:
            stream_logs(run_id=123, channel=mock_channel, duration=1)
            mock_print.assert_any_call("print_result_1")
            mock_print.assert_any_call("print_result_2")
            mock_print.assert_any_call("print_result_3")

    @patch("flwr.cli.log.ExecStub")
    def test_flwr_log_print_method(self, mock_stub: Mock) -> None:
        """Test print_logs."""
        # Create mock response iterator
        mock_response_iterator = iter(
            [StreamLogsResponse(log_output=f"stream_result_{i}") for i in range(1, 4)]
        )

        # Set up stub
        mock_stub_instance = mock_stub.return_value
        mock_stub_instance.StreamLogs.return_value = mock_response_iterator

        # Create mock channel
        mock_channel = Mock()

        with patch("builtins.print") as mock_print:
            print_logs(run_id=123, channel=mock_channel, timeout=0, is_test=True)
            mock_print.assert_any_call("stream_result_1")
            mock_print.assert_any_call("stream_result_2")
            mock_print.assert_any_call("stream_result_3")

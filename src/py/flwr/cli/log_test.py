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
from unittest.mock import MagicMock, patch

import grpc

from flwr.proto.exec_pb2 import StreamLogsRequest, StreamLogsResponse
from flwr.proto.exec_pb2_grpc import ExecStub

from . import client
from .client import run_client

# import client


# from . import client


# def run_client():
#     with grpc.insecure_channel("localhost:50051") as channel:
#         stub = ExecStub(channel)
#         req = StreamLogsRequest(run_id=123)
#
#         for res in stub.StreamLogs(req):
#             print(res.reply)


class TestLogStreamer(unittest.TestCase):
    @patch("client.grpc.insecure_channel")
    def test_print(self, mock_insecure_channel):
        # Mock the gRPC channel and stub
        mock_channel = MagicMock()
        mock_insecure_channel.return_value.__enter__.return_value = mock_channel

        # Create a mock response stream
        response1 = StreamLogsResponse(log_output="Reply 1")
        response2 = StreamLogsResponse(log_output="Reply 1")
        mock_stream = iter([response1, response2])

        # Set up the stub to return the mock stream
        mock_stub = MagicMock()
        mock_stub.StreamLogs.return_value = mock_stream
        mock_channel.ExecStub.return_value = mock_stub

        # Capture print output
        with patch("builtins.print") as mocked_print:
            run_client()
            mocked_print.assert_any_call("Client received: Reply 1")
            mocked_print.assert_any_call("Client received: Reply 1")


if __name__ == "__main__":
    unittest.main()

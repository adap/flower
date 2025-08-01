# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for run_app_scheduler function."""


import unittest
from unittest.mock import Mock, patch

from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.supercore.scheduler.run_scheduler import run_app_scheduler


class MockError(Exception):
    """Custom exception for termination."""


# pylint: disable=too-many-instance-attributes
class TestRunAppScheduler(unittest.TestCase):
    """Test the run_app_scheduler function."""

    def setUp(self) -> None:
        """Set up the test case."""
        # Patch the gRPC channel to accelerate testing
        self.mock_channel = Mock()
        self.patcher_channel = patch(
            "flwr.supercore.scheduler.run_scheduler.create_channel",
            return_value=self.mock_channel,
        )
        self.patcher_channel.start()

        # Patch ClientAppIoStub and assign to self.mock_stub
        self.mock_stub = Mock()
        self.patcher_stub = patch(
            "flwr.supercore.scheduler.run_scheduler.ClientAppIoStub",
            return_value=self.mock_stub,
        )
        self.patcher_stub.start()

        # Patch `time.sleep`
        self.mock_sleep = Mock()
        self.patcher_sleep = patch("time.sleep", self.mock_sleep)
        self.patcher_sleep.start()

        # Mock the plugin class
        self.mock_plugin = Mock()
        self.mock_plugin.select_run_id.return_value = 999
        self.mock_plugin_class = Mock(return_value=self.mock_plugin)

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.patcher_channel.stop()
        self.patcher_stub.stop()
        self.patcher_sleep.stop()

    def test_get_run_callable(self) -> None:
        """Test that run_app_scheduler creates the `get_run` callable correctly."""
        # Prepare
        self.mock_sleep.side_effect = MockError()
        self.mock_stub.GetRun.return_value = Mock(run=ProtoRun(run_id=110))

        # Execute
        with self.assertRaises(MockError):
            run_app_scheduler(
                plugin_class=self.mock_plugin_class,
                appio_api_address="1.2.3.4:1234",
                flwr_dir=None,
            )
        # Get the `get_run` callable from the plugin class
        _, kwargs = self.mock_plugin_class.call_args
        get_run_callable = kwargs["get_run"]
        run = get_run_callable(110)
        request_sent = self.mock_stub.GetRun.call_args[0][0]

        # Assert
        self.mock_stub.GetRun.assert_called_once()
        self.assertIsInstance(request_sent, GetRunRequest)
        self.assertEqual(request_sent.run_id, 110)
        self.assertEqual(run.run_id, 110)

    def test_no_run_ids(self) -> None:
        """Test that run_app_scheduler handles no run IDs gracefully."""
        # Prepare
        self.mock_sleep.side_effect = MockError()
        self.mock_stub.GetRunIdsWithPendingMessages.return_value = Mock(run_ids=[])

        # Execute
        with self.assertRaises(MockError):
            run_app_scheduler(
                plugin_class=self.mock_plugin_class,
                appio_api_address="1.2.3.4:1234",
                flwr_dir=None,
            )

        # Assert
        self.mock_stub.GetRunIdsWithPendingMessages.assert_called_once()
        self.mock_stub.RequestToken.assert_not_called()
        self.mock_plugin.select_run_id.assert_not_called()
        self.mock_plugin.launch_app.assert_not_called()

    def test_token_not_granted(self) -> None:
        """Test that run_app_scheduler does not launch app if token is not granted."""
        # Prepare
        self.mock_sleep.side_effect = MockError()
        self.mock_stub.GetRunIdsWithPendingMessages.return_value = Mock(run_ids=[110])
        self.mock_stub.RequestToken.return_value = Mock(token=b"")

        # Execute
        with self.assertRaises(MockError):
            run_app_scheduler(
                plugin_class=self.mock_plugin_class,
                appio_api_address="1.2.3.4:1234",
                flwr_dir=None,
            )

        # Assert
        self.mock_plugin.select_run_id.assert_called_once_with(candidate_run_ids=[110])
        self.mock_stub.RequestToken.assert_called_once()
        self.mock_plugin.launch_app.assert_not_called()

    def test_token_granted(self) -> None:
        """Test that run_app_scheduler launches app if token is granted (normal
        case)."""
        # Prepare
        fake_token = b"I am a valid token"
        run_id = 120
        self.mock_sleep.side_effect = MockError()
        self.mock_stub.GetRunIdsWithPendingMessages.return_value = Mock(
            run_ids=[run_id]
        )
        self.mock_plugin.select_run_id.return_value = run_id
        self.mock_stub.RequestToken.return_value = Mock(token=fake_token)

        # Execute
        with self.assertRaises(MockError):
            run_app_scheduler(
                plugin_class=self.mock_plugin_class,
                appio_api_address="1.2.3.4:1234",
                flwr_dir=None,
            )

        # Assert
        self.mock_plugin.select_run_id.assert_called_once_with(
            candidate_run_ids=[run_id]
        )
        self.mock_plugin.launch_app.assert_called_once_with(
            token=fake_token, run_id=run_id
        )

    def test_channel_close_on_exit(self) -> None:
        """Test that the channel is closed on exit."""
        # Prepare
        self.mock_sleep.side_effect = MockError()

        # Execute
        with self.assertRaises(MockError):
            run_app_scheduler(
                plugin_class=self.mock_plugin_class,
                appio_api_address="1.2.3.4:1234",
                flwr_dir=None,
            )

        # Assert
        self.mock_channel.close.assert_called_once()

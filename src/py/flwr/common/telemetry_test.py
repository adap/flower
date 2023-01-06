# Copyright 2023 Adap GmbH. All Rights Reserved.
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
"""Telemetry tests."""

import unittest
from io import StringIO
from unittest import mock

from flwr.common import telemetry


class TelemetryTest(unittest.TestCase):
    """Tests for the telemetry module."""

    def test_send(self) -> None:
        """Test if sending works against the actual API."""
        # Prepare
        expected = "{'status': 'created'}"

        # Execute
        actual = telemetry.send(event_type=telemetry.EventType.START_SERVER)

        # Assert
        self.assertEqual(actual, expected)

    @mock.patch("flwr.common.telemetry.FLWR_TELEMETRY_ENABLED", "0")
    def test_no_send(self) -> None:
        """Test if disableing sending works."""
        # Prepare
        expected = ""

        # Execute
        actual = telemetry.send(event_type=telemetry.EventType.START_SERVER)

        # Assert
        self.assertEqual(actual, expected)

    @mock.patch("flwr.common.telemetry.FLWR_TELEMETRY_ENABLED", "0")
    @mock.patch("flwr.common.telemetry.FLWR_TELEMETRY_LOGGING", "1")
    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_logging(self, stdout: StringIO) -> None:
        """Test if logging works.

        NOTE: Sending is disabled as we don't need it to test.
        """
        # Prepare
        expected_return = ""
        expected_stdout = "POST"  # Just checking for a substring

        # Execute
        actual_return = telemetry.send(event_type=telemetry.EventType.START_SERVER)

        # Assert
        self.assertEqual(actual_return, expected_return)
        self.assertIn(expected_stdout, stdout.getvalue())

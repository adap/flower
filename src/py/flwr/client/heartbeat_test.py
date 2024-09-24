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
"""Unit tests for heartbeat utility functions."""


import threading
import time
import unittest
from unittest.mock import MagicMock

from .heartbeat import start_ping_loop


class TestStartPingLoopWithFailures(unittest.TestCase):
    """Test heartbeat utility functions."""

    def test_ping_loop_terminates(self) -> None:
        """Test if the ping loop thread terminates when flagged."""
        # Prepare
        ping_fn = MagicMock()
        stop_event = threading.Event()

        # Execute
        thread = start_ping_loop(ping_fn, stop_event)
        time.sleep(1)
        stop_event.set()
        thread.join(timeout=1)

        # Assert
        self.assertTrue(ping_fn.called)
        self.assertFalse(thread.is_alive())

    def test_ping_loop_with_failures_terminates(self) -> None:
        """Test if the ping loop thread with failures terminates when flagged."""
        # Prepare
        ping_fn = MagicMock(side_effect=RuntimeError())
        stop_event = threading.Event()

        # Execute
        thread = start_ping_loop(ping_fn, stop_event)
        time.sleep(1)
        stop_event.set()
        thread.join(timeout=1)

        # Assert
        self.assertTrue(ping_fn.called)
        self.assertFalse(thread.is_alive())

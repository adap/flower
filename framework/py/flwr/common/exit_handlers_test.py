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
"""Tests for exit handler utils."""


import os
import signal
import unittest
from unittest.mock import Mock, patch

from .exit_handlers import (
    add_exit_handler,
    register_exit_handlers,
    registered_exit_handlers,
)
from .telemetry import EventType


class TestExitHandlers(unittest.TestCase):
    """Tests for exit handler utils."""

    def setUp(self) -> None:
        """Clear all exit handlers before each test."""
        registered_exit_handlers.clear()

    @patch("sys.exit")
    def test_register_exit_handlers(self, mock_sys_exit: Mock) -> None:
        """Test register_exit_handlers."""
        # Prepare
        handlers = [Mock(), Mock(), Mock()]
        register_exit_handlers(EventType.PING, exit_handlers=handlers[:-1])  # type: ignore
        add_exit_handler(handlers[-1])

        # Execute
        os.kill(os.getpid(), signal.SIGTERM)

        # Assert
        for handler in handlers:
            handler.assert_called()
        mock_sys_exit.assert_called()
        self.assertEqual(registered_exit_handlers, handlers)

    def test_add_exit_handler(self) -> None:
        """Test add_exit_handler."""
        # Prepare
        handler = Mock()

        # Execute
        add_exit_handler(handler)

        # Assert
        self.assertIn(handler, registered_exit_handlers)

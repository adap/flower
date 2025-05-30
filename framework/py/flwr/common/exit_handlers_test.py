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
    _handlers,
    add_exit_handler,
    register_exit_handlers,
    remove_exit_handler,
)
from .telemetry import EventType


class TestExitHandlers(unittest.TestCase):
    """Tests for exit handler utils."""

    def setUp(self) -> None:
        """Clear all exit handlers before each test."""
        _handlers.clear()

    @patch("sys.exit")
    def test_register_exit_handlers(self, mock_sys_exit: Mock) -> None:
        """Test register_exit_handlers."""
        # Prepare
        handlers = [Mock(), Mock()]
        register_exit_handlers(EventType.PING, handlers=handlers)  # type: ignore

        # Execute
        os.kill(os.getpid(), signal.SIGTERM)

        # Assert
        for handler in handlers:
            handler.assert_called()
        mock_sys_exit.assert_called()
        self.assertEqual(list(_handlers.values()), handlers)

    def test_add_exit_handler(self) -> None:
        """Test add_exit_handler."""
        # Prepare
        handler = Mock()

        # Execute
        add_exit_handler(handler, "mock_handler")

        # Assert
        self.assertIn("mock_handler", _handlers)
        self.assertEqual(_handlers["mock_handler"], handler)

    def test_remove_exit_handler(self) -> None:
        """Test remove_exit_handler."""
        # Prepare
        handler = Mock()
        add_exit_handler(handler, "mock_handler")

        # Execute
        remove_exit_handler("mock_handler")

        # Assert
        self.assertNotIn("mock_handler", _handlers)

    def test_remove_exit_handler_not_found(self) -> None:
        """Test remove_exit_handler with invalid name."""
        # Prepare
        handler = Mock()
        add_exit_handler(handler, "mock_handler")

        # Execute
        with self.assertRaises(KeyError):
            remove_exit_handler("non_existent_handler")

        # Assert
        self.assertIn("mock_handler", _handlers)
        self.assertEqual(_handlers["mock_handler"], handler)

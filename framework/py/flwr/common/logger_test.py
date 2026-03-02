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
"""Flower Logger tests."""


import logging
import sys
from queue import Queue
from typing import Any

from . import logger as logger_module
from .logger import ConsoleHandler, mirror_output_to_queue, restore_output


def test_mirror_output_to_queue() -> None:
    """Test that stdout and stderr are mirrored to the provided queue."""
    # Prepare
    log_queue: Queue[str | None] = Queue()

    # Execute
    mirror_output_to_queue(log_queue)
    print("Test message")
    sys.stderr.write("Error message\n")

    # Assert
    assert not log_queue.empty()
    assert log_queue.get() == "Test message"
    assert log_queue.get() == "\n"
    assert log_queue.get() == "Error message\n"


def test_restore_output() -> None:
    """Test that stdout and stderr are restored after calling restore_output."""
    # Prepare
    log_queue: Queue[str | None] = Queue()

    # Execute
    mirror_output_to_queue(log_queue)
    print("Test message before restore")
    restore_output()
    print("Test message after restore")
    sys.stderr.write("Error message after restore\n")

    # Assert
    assert log_queue.get() == "Test message before restore"
    assert log_queue.get() == "\n"
    assert log_queue.empty()


def test_console_handler_rebuilds_console_for_streamed_colored_logs(
    monkeypatch: Any,
) -> None:
    """Test that streamed logs keep color by recreating Console on stream change."""
    inits: list[dict[str, Any]] = []

    class DummyConsole:
        """Capture Console init kwargs and ignore writes."""

        def __init__(self, **kwargs: Any) -> None:
            inits.append(kwargs)

        def print(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(logger_module, "Console", DummyConsole)
    handler = ConsoleHandler(colored=True)

    handler.emit(logging.LogRecord("flwr", logging.INFO, "", 0, "first", (), None))
    handler.stream = sys.stdout
    handler.emit(logging.LogRecord("flwr", logging.INFO, "", 0, "second", (), None))

    assert len(inits) == 2
    assert all(kwargs["force_terminal"] is True for kwargs in inits)
    assert all(kwargs["no_color"] is False for kwargs in inits)

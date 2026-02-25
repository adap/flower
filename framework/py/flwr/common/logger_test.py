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
from io import StringIO
from queue import Queue

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


class _FakeStream(StringIO):
    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_console_handler_disables_ansi_for_non_tty_stream() -> None:
    """Test that ANSI escape codes are not emitted for non-TTY streams."""
    handler = ConsoleHandler(colored=True, stream=_FakeStream(is_tty=False))
    record = logging.LogRecord(
        name="flwr",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )

    formatted = handler.format(record)

    assert "\033[" not in formatted
    assert formatted.startswith("INFO")

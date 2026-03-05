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
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from queue import Queue

from .logger import (
    FLOWER_LOGGER,
    configure_superlink_log_file,
    mirror_output_to_queue,
    restore_output,
)


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


def test_configure_superlink_log_file(tmp_path: Path) -> None:
    """Test configuring timed file rotation for SuperLink logs."""
    # Prepare
    file_name = tmp_path / "test-superlink.log"
    path = file_name.resolve()
    before = list(FLOWER_LOGGER.handlers)

    try:
        # Execute
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )

        # Assert
        handler = next(
            (
                h
                for h in FLOWER_LOGGER.handlers
                if isinstance(h, TimedRotatingFileHandler)
                and Path(h.baseFilename).resolve() == path
            ),
            None,
        )
        assert handler is not None
        assert handler.level == logging.DEBUG
        assert handler.backupCount == 7
        assert handler.interval == 24 * 60 * 60
    finally:
        # Clean up any handlers introduced by this test
        for handler in list(FLOWER_LOGGER.handlers):
            if handler in before:
                continue
            FLOWER_LOGGER.removeHandler(handler)
            handler.close()


def test_configure_superlink_log_file_idempotent(tmp_path: Path) -> None:
    """Test configuring SuperLink rotation twice does not duplicate handlers."""
    # Prepare
    file_name = tmp_path / "test-superlink-idempotent.log"
    path = file_name.resolve()
    before = list(FLOWER_LOGGER.handlers)

    try:
        # Execute
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )

        # Assert
        handlers = [
            h
            for h in FLOWER_LOGGER.handlers
            if isinstance(h, TimedRotatingFileHandler)
            and Path(h.baseFilename).resolve() == path
        ]
        assert len(handlers) == 1
    finally:
        for handler in list(FLOWER_LOGGER.handlers):
            if handler in before:
                continue
            FLOWER_LOGGER.removeHandler(handler)
            handler.close()

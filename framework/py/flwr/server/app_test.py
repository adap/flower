# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for Flower SuperLink app CLI argument parsing."""


import pytest

from .app import _parse_args_run_superlink


def test_parse_superlink_log_rotation_args_defaults() -> None:
    """SuperLink log rotation args should have expected defaults."""
    # Execute
    args = _parse_args_run_superlink().parse_args([])

    # Assert
    assert args.log_file is None
    assert args.log_rotation_interval_hours == 24
    assert args.log_rotation_backup_count == 7


def test_parse_superlink_log_rotation_args_custom_values() -> None:
    """SuperLink log rotation args should parse explicit values."""
    # Execute
    args = _parse_args_run_superlink().parse_args(
        [
            "--log-file",
            "/tmp/superlink.log",
            "--log-rotation-interval-hours",
            "12",
            "--log-rotation-backup-count",
            "14",
        ]
    )

    # Assert
    assert args.log_file == "/tmp/superlink.log"
    assert args.log_rotation_interval_hours == 12
    assert args.log_rotation_backup_count == 14


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_superlink_log_rotation_interval_requires_positive_int(
    value: str,
) -> None:
    """The interval must be a positive integer."""
    with pytest.raises(SystemExit):
        _parse_args_run_superlink().parse_args(
            ["--log-rotation-interval-hours", value]
        )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_superlink_log_rotation_backup_requires_positive_int(
    value: str,
) -> None:
    """The backup count must be a positive integer."""
    with pytest.raises(SystemExit):
        _parse_args_run_superlink().parse_args(
            ["--log-rotation-backup-count", value]
        )

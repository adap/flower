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
"""Unified exit function."""


from __future__ import annotations

import sys
from logging import ERROR, INFO
from typing import NoReturn

from flwr.common import EventType, event

from .logger import log


class ExitCode:
    """Exit codes for Flower components."""

    # System exit codes
    SUCCESS = 0
    GENERIC_ERROR = 1
    GRACEFUL_EXIT = 10  # Graceful exit requested by the user

    # SuperLink-specific exit codes (100-199)
    THREAD_CRASH = 100

    # ServerApp-specific exit codes (200-299)

    # SuperNode-specific exit codes (300-399)
    REST_ADDRESS_INVALID = 300
    NODE_AUTH_KEYS_REQUIRED = 301
    NODE_AUTH_KEYS_INVALID = 302

    # ClientApp-specific exit codes (400-499)

    # Common exit codes (500-999)
    IP_ADDRESS_INVALID = 500
    MISSING_EXTRA_REST = 501
    TLS_NOT_SUPPORTED = 502

    # Deprecated exit codes (1000-)
    DEPRECATED_APP_ARGUMENT = 1000  # `flower-supernode <app>` is deprecated

    def __new__(cls) -> ExitCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


HELP_PAGE_URL = "https://flower.ai/docs/framework/ref-exit-code"
KNOWN_EXIT_CODE_HELP: set[int] = set()


def flwr_exit(code: int, message: str | None = None) -> NoReturn:
    """Handle application exit with an optional message."""
    # Construct exit message
    exit_message = f"Exit Code: {code}"
    if message:
        exit_message += f"\n{message}"

    # Set log level and system exit code
    log_level = INFO
    sys_exit_code = 0
    if code not in {ExitCode.SUCCESS, ExitCode.GRACEFUL_EXIT}:
        log_level = ERROR
        sys_exit_code = 1

    # Add help URL for known exit codes
    if code in KNOWN_EXIT_CODE_HELP:
        help_url = f"{HELP_PAGE_URL}.e{code}.html"
        exit_message += f"\n\nFor more information, visit: <{help_url}>"

    # Log the exit message
    log(log_level, exit_message)

    # Telemetry event
    event_type = _try_obtain_telemetry_event()
    if event_type:
        event(event_type, event_details={"exit_code": code, "message": message})

    # Exit
    sys.exit(sys_exit_code)


def _try_obtain_telemetry_event() -> EventType | None:
    """Try to obtain a telemetry event."""
    if sys.argv[0].endswith("flower-superlink"):
        return EventType.RUN_SUPERLINK_LEAVE
    if sys.argv[0].endswith("flower-supernode"):
        return EventType.RUN_SUPERNODE_LEAVE
    if sys.argv[0].endswith("flwr-serverapp"):
        return EventType.FLWR_SERVERAPP_RUN_LEAVE
    if sys.argv[0].endswith("flwr-clientapp"):
        return None  # Not yet implemented
    return None

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
from typing import Any, NoReturn

from flwr.common import EventType, event

from ..logger import log
from .exit_code import EXIT_CODE_HELP

HELP_PAGE_URL = "https://flower.ai/docs/framework/ref-exit-codes/"


def flwr_exit(
    code: int,
    message: str | None = None,
    event_type: EventType | None = None,
    event_details: dict[str, Any] | None = None,
) -> NoReturn:
    """Handle application exit with an optional message.

    The exit message logged and displayed will follow this structure:

    >>> Exit Code: <code>
    >>> <message>
    >>> <short-help-message>
    >>>
    >>> For more information, visit: <help-page-url>

    - `<code>`: The unique exit code representing the termination reason.
    - `<message>`: Optional context or additional information about the exit.
    - `<short-help-message>`: A brief explanation for the given exit code.
    - `<help-page-url>`: A URL providing detailed documentation and resolution steps.
    """
    is_error = not 0 <= code < 100  # 0-99 are success exit codes

    # Construct exit message
    exit_message = f"Exit Code: {code}\n" if is_error else ""
    exit_message += message or ""
    if short_help_message := EXIT_CODE_HELP.get(code, ""):
        exit_message += f"\n{short_help_message}"

    # Set log level and system exit code
    log_level = ERROR if is_error else INFO
    sys_exit_code = 1 if is_error else 0

    # Add help URL for non-successful/graceful exits
    if is_error:
        help_url = f"{HELP_PAGE_URL}{code}.html"
        exit_message += f"\n\nFor more information, visit: <{help_url}>"

    # Telemetry event
    event_type = event_type or _try_obtain_telemetry_event()
    if event_type:
        event_details = event_details or {}
        event_details["exit_code"] = code
        event(event_type, event_details).result()

    # Log the exit message
    log(log_level, exit_message)

    # Exit
    sys.exit(sys_exit_code)


# pylint: disable-next=too-many-return-statements
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
    if sys.argv[0].endswith("flwr-simulation"):
        return EventType.FLWR_SIMULATION_RUN_LEAVE
    if sys.argv[0].endswith("flower-simulation"):
        return EventType.CLI_FLOWER_SIMULATION_LEAVE
    return None

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
    SUPERLINK_THREAD_CRASH = 100

    # ServerApp-specific exit codes (200-299)

    # SuperNode-specific exit codes (300-399)
    SUPERNODE_REST_ADDRESS_INVALID = 300
    SUPERNODE_NODE_AUTH_KEYS_REQUIRED = 301
    SUPERNODE_NODE_AUTH_KEYS_INVALID = 302

    # ClientApp-specific exit codes (400-499)

    # Common exit codes (500-899)
    COMMON_ADDRESS_INVALID = 500
    COMMON_MISSING_EXTRA_REST = 501
    COMMON_TLS_NOT_SUPPORTED = 502

    # Deprecated exit codes (900-)
    SUPERNODE_REMOVED_APP_ARGUMENT = 900  # Deprecated `flower-supernode <app>`

    def __new__(cls) -> ExitCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


# All short help messages for exit codes
EXIT_CODE_HELP = {
    # System exit codes
    ExitCode.SUCCESS: "",
    ExitCode.GENERIC_ERROR: "",
    ExitCode.GRACEFUL_EXIT: "",
    # SuperLink-specific exit codes (100-199)
    ExitCode.SUPERLINK_THREAD_CRASH: "An important background thread has crashed.",
    # ServerApp-specific exit codes (200-299)
    # SuperNode-specific exit codes (300-399)
    ExitCode.SUPERNODE_REST_ADDRESS_INVALID: (
        "When using the REST API, please provide `https://` or "
        "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
    ),
    ExitCode.SUPERNODE_NODE_AUTH_KEYS_REQUIRED: (
        "Node authentication requires file paths to both "
        "'--auth-supernode-private-key' and '--auth-supernode-public-key' "
        "to be provided (providing only one of them is not sufficient)."
    ),
    ExitCode.SUPERNODE_NODE_AUTH_KEYS_INVALID: (
        "Node uthentication requires elliptic curve private and public key pair. "
        "Please ensure that the file path points to a valid private/public key "
        "file and try again."
    ),
    # ClientApp-specific exit codes (400-499)
    # Common exit codes (500-999)
    ExitCode.COMMON_ADDRESS_INVALID: (
        "Please provide a valid URL, IPv4 or IPv6 address."
    ),
    ExitCode.COMMON_MISSING_EXTRA_REST: """
Extra dependencies required for using the REST-based Fleet API are missing.

To use the REST API, install `flwr` with the `rest` extra:

    `pip install "flwr[rest]"`.
""",
    ExitCode.COMMON_TLS_NOT_SUPPORTED: "Please use the '--insecure' flag.",
    # Deprecated exit codes (1000-)
    ExitCode.SUPERNODE_REMOVED_APP_ARGUMENT: (
        "The `app` argument has been removed. "
        "Please remove the `app` argument from your command."
    ),
}


HELP_PAGE_URL = "https://flower.ai/docs/framework/exit-codes/"


def flwr_exit(
    code: int, message: str | None = None, event_type: EventType | None = None
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
    is_error = code not in {ExitCode.SUCCESS, ExitCode.GRACEFUL_EXIT}

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
        event(event_type, event_details={"exit_code": code}).result()

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

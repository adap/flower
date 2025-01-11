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
from typing import Optional
from logging import INFO, ERROR
from .logger import log
import sys



class ErrorCode:
    """Error codes for Message's Error."""

    # System exit codes
    SUCCESS = 0
    GENERIC_ERROR = 1
    GRACEFUL_EXIT = 10  # Graceful exit requested by the user
    
    # SuperLink-specific exit codes (100-199)
    THREAD_CRASH = 100

    # ServerApp-specific exit codes (200-299)
    
    # SuperNode-specific exit codes (300-399)
    AUTH_KEYS_REQUIRED = 300
    
    # ClientApp-specific exit codes (400-499)
    
    # Common exit codes (500-699)
    ADDRESS_CANNOT_BE_PARSED = 500
    MISSING_EXTRA_REST = 501
    
    # Deprecated exit codes (700-799)
    DEPRECATED_APP_ARGUMENT = 700  # `flower-supernode <app>` is deprecated
    


    def __new__(cls) -> ErrorCode:
        """Prevent instantiation."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


HELP_PAGE_URL = "https://flower.ai/docs/framework/ref-exit-code."


def exit(code: int, message: Optional[str] = None) -> None:
    """Handle application exit with an optional message."""
    exit_message = f"Exit Code: {code}"
    if message:
        exit_message += f"\\n\\n{message}"

    log_level = INFO
    sys_exit_code = 0

    if code not in {ErrorCode.SUCCESS, ErrorCode.GRACEFUL_EXIT}:
        exit_message += f"\\n\\nFor more information, visit: <{HELP_PAGE_URL}{code}.html>"
        log_level = ERROR
        sys_exit_code = 1

    log(log_level, exit_message)

    # Placeholder for additional telemetry (e.g., sending exit data)
    # Example: telemetry.send_exit_event(code, message)

    sys.exit(sys_exit_code)



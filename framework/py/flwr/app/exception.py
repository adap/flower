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
"""Flower application exceptions."""


class AppExitException(BaseException):
    """Base exception for all application-level errors in ServerApp and ClientApp.

    When raised, the process will exit and report a telemetry event with the associated
    exit code. This is not intended to be caught by user code.
    """

    # Default exit code — subclasses must override
    exit_code = -1

    def __init_subclass__(cls) -> None:
        """Ensure subclasses override the exit_code attribute."""
        if cls.exit_code == -1:
            raise ValueError("Subclasses must override the exit_code attribute.")

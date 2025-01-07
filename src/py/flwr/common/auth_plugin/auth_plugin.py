# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Abstract classes for Flower User Auth Plugin."""


from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from flwr.proto.exec_pb2_grpc import ExecStub


class ExecAuthPlugin(ABC):
    """Abstract Flower Auth Plugin class for ExecServicer.

    Parameters
    ----------
    config : dict[str, Any]
        The authentication configuration loaded from a YAML file.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """Abstract constructor."""

    @abstractmethod
    def get_login_details(self) -> dict[str, str]:
        """Get the login details."""

    @abstractmethod
    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> bool:
        """Validate authentication tokens in the provided metadata."""

    @abstractmethod
    def get_auth_tokens(self, auth_details: dict[str, str]) -> dict[str, str]:
        """Get authentication tokens."""

    @abstractmethod
    def refresh_tokens(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Optional[Sequence[tuple[str, Union[str, bytes]]]]:
        """Refresh authentication tokens in the provided metadata."""


class CliAuthPlugin(ABC):
    """Abstract Flower Auth Plugin class for CLI.

    Parameters
    ----------
    user_auth_config_path : Path
        The path to the user's authentication configuration file.
    """

    @staticmethod
    @abstractmethod
    def login(
        login_details: dict[str, str],
        exec_stub: ExecStub,
    ) -> dict[str, Any]:
        """Authenticate the user with the SuperLink.

        Parameters
        ----------
        login_details : dict[str, str]
            A dictionary containing the user's login details.
        exec_stub : ExecStub
            An instance of `ExecStub` used for communication with the SuperLink.

        Returns
        -------
        user_auth_config : dict[str, Any]
            A dictionary containing the user's authentication configuration
            in JSON format.
        """

    @abstractmethod
    def __init__(self, user_auth_config_path: Path):
        """Abstract constructor."""

    @abstractmethod
    def store_tokens(self, user_auth_config: dict[str, Any]) -> None:
        """Store authentication tokens from the provided user_auth_config.

        The configuration, including tokens, will be saved as a JSON file
        at `user_auth_config_path`.
        """

    @abstractmethod
    def load_tokens(self) -> None:
        """Load authentication tokens from the user_auth_config_path."""

    @abstractmethod
    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Sequence[tuple[str, Union[str, bytes]]]:
        """Write authentication tokens to the provided metadata."""

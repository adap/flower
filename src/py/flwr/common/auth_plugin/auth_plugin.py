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

from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub


class ExecAuthPlugin(ABC):
    """Abstract Flower Auth Plugin class for ExecServicer."""

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """Abstract constructor (init)."""

    @abstractmethod
    def get_login_details(self) -> GetLoginDetailsResponse:
        """Get the GetLoginDetailsResponse containing the login details."""

    @abstractmethod
    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> bool:
        """Validate the auth tokens in the provided metadata."""

    @abstractmethod
    def get_auth_tokens(self, request: GetAuthTokensRequest) -> GetAuthTokensResponse:
        """Get the relevant auth tokens."""

    @abstractmethod
    def refresh_tokens(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Optional[Sequence[tuple[str, Union[str, bytes]]]]:
        """Refresh auth tokens in the provided metadata."""


class CliAuthPlugin(ABC):
    """Abstract Flower Auth Plugin class for CLI."""

    @staticmethod
    @abstractmethod
    def login(
        login_details: dict[str, str],
        config: dict[str, Any],
        federation: str,
        exec_stub: ExecStub,
    ) -> dict[str, Any]:
        """Login logic to log in user to the SuperLink.

        Parameters
        ----------
        login_details : dict[str, str]
            A dictionary containing the login details.
        config : dict[str, Any]
            The project configuration.
        federation : str
            The name of the federation to which the user belongs.
        exec_stub : ExecStub
            An instance of `ExecStub` used for communication with the SuperLink.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the user's authentication details.
        """

    @abstractmethod
    def __init__(self, config: dict[str, Any], config_path: Path):
        """Abstract constructor (init)."""

    @abstractmethod
    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Sequence[tuple[str, Union[str, bytes]]]:
        """Write relevant auth tokens to the provided metadata."""

    @abstractmethod
    def store_tokens(self, config: dict[str, Any]) -> None:
        """Store the tokens from the provided config.

        The tokens will be saved as a JSON file in the provided config_path.
        """

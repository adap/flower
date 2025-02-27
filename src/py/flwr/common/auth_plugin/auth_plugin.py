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
from typing import Optional, Union

from flwr.common.typing import UserInfo
from flwr.proto.exec_pb2_grpc import ExecStub

from ..typing import UserAuthCredentials, UserAuthLoginDetails


class ExecAuthPlugin(ABC):
    """Abstract Flower Auth Plugin class for ExecServicer.

    Parameters
    ----------
    user_auth_config_path : Path
        Path to the YAML file containing the authentication configuration.
    """

    @abstractmethod
    def __init__(
        self,
        user_auth_config_path: Path,
        verify_tls_cert: bool,
    ):
        """Abstract constructor."""

    @abstractmethod
    def get_login_details(self) -> Optional[UserAuthLoginDetails]:
        """Get the login details."""

    @abstractmethod
    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> tuple[bool, Optional[UserInfo]]:
        """Validate authentication tokens in the provided metadata."""

    @abstractmethod
    def get_auth_tokens(self, device_code: str) -> Optional[UserAuthCredentials]:
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
    credentials_path : Path
        Path to the user's authentication credentials file.
    """

    @staticmethod
    @abstractmethod
    def login(
        login_details: UserAuthLoginDetails,
        exec_stub: ExecStub,
    ) -> UserAuthCredentials:
        """Authenticate the user and retrieve authentication credentials.

        Parameters
        ----------
        login_details : UserAuthLoginDetails
            An object containing the user's login details.
        exec_stub : ExecStub
            A stub for executing RPC calls to the server.

        Returns
        -------
        UserAuthCredentials
            The authentication credentials obtained after login.
        """

    @abstractmethod
    def __init__(self, credentials_path: Path):
        """Abstract constructor."""

    @abstractmethod
    def store_tokens(self, credentials: UserAuthCredentials) -> None:
        """Store authentication tokens to the `credentials_path`.

        The credentials, including tokens, will be saved as a JSON file
        at `credentials_path`.
        """

    @abstractmethod
    def load_tokens(self) -> None:
        """Load authentication tokens from the `credentials_path`."""

    @abstractmethod
    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Sequence[tuple[str, Union[str, bytes]]]:
        """Write authentication tokens to the provided metadata."""

    @abstractmethod
    def read_tokens_from_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Optional[UserAuthCredentials]:
        """Read authentication tokens from the provided metadata."""

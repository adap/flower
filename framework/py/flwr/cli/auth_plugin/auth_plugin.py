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
"""Abstract classes for Flower User Auth Plugin."""


from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from flwr.common.typing import UserAuthCredentials, UserAuthLoginDetails
from flwr.proto.control_pb2_grpc import ControlStub


class LoginError(Exception):
    """Login error exception."""

    def __init__(self, message: str):
        self.message = message


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
        control_stub: ControlStub,
    ) -> UserAuthCredentials:
        """Authenticate the user and retrieve authentication credentials.

        Parameters
        ----------
        login_details : UserAuthLoginDetails
            An object containing the user's login details.
        control_stub : ControlStub
            A stub for executing RPC calls to the server.

        Returns
        -------
        UserAuthCredentials
            The authentication credentials obtained after login.

        Raises
        ------
        LoginError
            If the login process fails.
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

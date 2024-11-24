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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import grpc

from flwr.proto.exec_pb2 import (
    GetAuthTokenRequest,
    GetAuthTokenResponse,
    LoginRequest,
    LoginResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

Metadata = List[Any]


class ExecAuthPlugin(ABC):
    """Abstract Flower Exec API Auth Plugin class."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def send_auth_endpoint(self) -> LoginResponse:
        """Send relevant login details as a LoginResponse."""
        pass

    @abstractmethod
    def authenticate(self, metadata: Sequence[Tuple[str, Union[str, bytes]]]):
        """Authenticate auth tokens in the provided metadata."""
        pass

    @abstractmethod
    def get_auth_token_response(
        self, request: GetAuthTokenRequest
    ) -> GetAuthTokenResponse:
        """Send relevant tokens as a GetAuthTokenResponse."""
        pass

    @abstractmethod
    def refresh_token(self, context: grpc.ServicerContext) -> Optional[Tuple[str, str]]:
        """Refresh auth tokens in the provided metadata."""
        pass


class UserAuthPlugin(ABC):
    """Abstract Flower User Auth Plugin class."""

    @staticmethod
    @abstractmethod
    def login(
        login_details: Dict[str, str],
        config: Dict[str, Any],
        federation: str,
        exec_stub: ExecStub,
    ):
        """Read relevant auth details from federation config."""
        pass

    @abstractmethod
    def __init__(self, config: Dict[str, Any], config_path: Path):
        """Abstract constructor (init)"""
        pass

    @abstractmethod
    def provide_auth_details(self, metadata) -> Metadata:
        """Provide relevant auth tokens in the metadata."""
        pass

    @abstractmethod
    def save_refreshed_auth_tokens(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> None:
        """Provide refreshed auth tokens to the config file."""
        pass

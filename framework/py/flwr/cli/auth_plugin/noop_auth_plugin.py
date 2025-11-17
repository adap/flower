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
"""Concrete NoOp implementation for CLI-side account authentication plugin."""


from collections.abc import Sequence
from pathlib import Path

from flwr.common.typing import AccountAuthCredentials, AccountAuthLoginDetails
from flwr.proto.control_pb2_grpc import ControlStub

from .auth_plugin import CliAuthPlugin, LoginError


class NoOpCliAuthPlugin(CliAuthPlugin):
    """No-operation implementation of the CliAuthPlugin."""

    @staticmethod
    def login(
        login_details: AccountAuthLoginDetails,
        control_stub: ControlStub,
    ) -> AccountAuthCredentials:
        """Raise LoginError as no-op plugin does not support login."""
        raise LoginError("Account authentication is not enabled on this SuperLink.")

    def __init__(self, credentials_path: Path) -> None:
        pass

    def store_tokens(self, credentials: AccountAuthCredentials) -> None:
        """Do nothing."""

    def load_tokens(self) -> None:
        """Do nothing."""

    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> Sequence[tuple[str, str | bytes]]:
        """Return the metadata unchanged."""
        return metadata

    def read_tokens_from_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> AccountAuthCredentials | None:
        """Return None."""
        return None

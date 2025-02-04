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
"""Flower CLI user auth plugin for OIDC."""


import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import typer

from flwr.common.auth_plugin import CliAuthPlugin
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTH_TYPE_JSON_KEY,
    REFRESH_TOKEN_KEY,
    AuthType,
)
from flwr.common.typing import UserAuthCredentials, UserAuthLoginDetails
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub


class OidcCliPlugin(CliAuthPlugin):
    """Flower OIDC auth plugin for CLI."""

    def __init__(self, credentials_path: Path):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.credentials_path = credentials_path

    @staticmethod
    def login(
        login_details: UserAuthLoginDetails,
        exec_stub: ExecStub,
    ) -> UserAuthCredentials:
        """Authenticate the user and retrieve authentication credentials."""
        typer.secho(
            "Please login with your user credentials here: "
            f"{login_details.verification_uri_complete}",
            fg=typer.colors.BLUE,
        )
        start_time = time.time()
        time.sleep(login_details.interval)

        while (time.time() - start_time) < login_details.expires_in:
            res: GetAuthTokensResponse = exec_stub.GetAuthTokens(
                GetAuthTokensRequest(device_code=login_details.device_code)
            )

            access_token = res.access_token
            refresh_token = res.refresh_token

            if access_token and refresh_token:
                typer.secho(
                    "✅ Login successful.",
                    fg=typer.colors.GREEN,
                    bold=False,
                )
                return UserAuthCredentials(
                    access_token=access_token,
                    refresh_token=refresh_token,
                )

            time.sleep(login_details.interval)

        typer.secho(
            "❌ Timeout, failed to sign in.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    def store_tokens(self, credentials: UserAuthCredentials) -> None:
        """Store authentication tokens to the `credentials_path`.

        The credentials, including tokens, will be saved as a JSON file
        at `credentials_path`.
        """
        self.access_token = credentials.access_token
        self.refresh_token = credentials.refresh_token
        json_dict = {
            AUTH_TYPE_JSON_KEY: AuthType.OIDC,
            ACCESS_TOKEN_KEY: credentials.access_token,
            REFRESH_TOKEN_KEY: credentials.refresh_token,
        }

        with open(self.credentials_path, "w", encoding="utf-8") as file:
            json.dump(json_dict, file, indent=4)

    def load_tokens(self) -> None:
        """Load authentication tokens from the `credentials_path`."""
        with open(self.credentials_path, encoding="utf-8") as file:
            json_dict: dict[str, Any] = json.load(file)
            access_token = json_dict.get(ACCESS_TOKEN_KEY)
            refresh_token = json_dict.get(REFRESH_TOKEN_KEY)

        if isinstance(access_token, str) and isinstance(refresh_token, str):
            self.access_token = access_token
            self.refresh_token = refresh_token

    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Sequence[tuple[str, Union[str, bytes]]]:
        """Write authentication tokens to the provided metadata."""
        if self.access_token is None or self.refresh_token is None:
            typer.secho(
                "❌ Missing authentication tokens. Please login first.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

        return list(metadata) + [
            (ACCESS_TOKEN_KEY, self.access_token),
            (REFRESH_TOKEN_KEY, self.refresh_token),
        ]

    def read_tokens_from_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Optional[UserAuthCredentials]:
        """Read authentication tokens from the provided metadata."""
        metadata_dict = dict(metadata)
        access_token = metadata_dict.get(ACCESS_TOKEN_KEY)
        refresh_token = metadata_dict.get(REFRESH_TOKEN_KEY)

        if isinstance(access_token, str) and isinstance(refresh_token, str):
            return UserAuthCredentials(
                access_token=access_token,
                refresh_token=refresh_token,
            )

        return None

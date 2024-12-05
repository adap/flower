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
"""Flower User Auth Plugin for Keycloak."""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union, cast

import grpc
import typer
from requests import post

from flwr.common.auth_plugin import CliAuthPlugin, ExecAuthPlugin
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from .constant import (
    ACCESS_TOKEN,
    AUTH_TYPE,
    AUTH_URL,
    CLIENT_ID,
    CLIENT_SECRET,
    DEVICE_CODE,
    DEVICE_FLOW_GRANT_TYPE,
    EXPIRES_IN,
    GRANT_TYPE,
    INTERVAL,
    REFRESH_TOKEN,
    TOKEN_URL,
    VALIDATE_URL,
    VERIFICATION_URI_COMPLETE,
)


class KeycloakExecPlugin(ExecAuthPlugin):
    """Flower Keycloak Auth Plugin for ExecServicer."""

    def __init__(self, config: dict[str, Any]):
        self.auth_url: str = config.get(AUTH_URL, "")
        self.token_url: str = config.get(TOKEN_URL, "")
        self.keycloak_client_id: str = config.get(CLIENT_ID, "")
        self.keycloak_client_secret: str = config.get(CLIENT_SECRET, "")
        self.validate_url: str = config.get(VALIDATE_URL, "")

    def get_login_details(self) -> GetLoginDetailsResponse:
        """Get the GetLoginDetailsResponse containing the login details."""
        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.auth_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            device_code = str(data[DEVICE_CODE])
            verification_uri_complete = str(data[VERIFICATION_URI_COMPLETE])
            expires_in = str(data[EXPIRES_IN])
            interval = str(data[INTERVAL])
            login_details = {
                AUTH_TYPE: "keycloak",
                DEVICE_CODE: device_code,
                VERIFICATION_URI_COMPLETE: verification_uri_complete,
                EXPIRES_IN: expires_in,
                INTERVAL: interval,
            }
            return GetLoginDetailsResponse(login_details=login_details)

        return GetLoginDetailsResponse(login_details={})

    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> bool:
        """Validate the auth tokens in the provided metadata."""
        metadata_dict = dict(metadata)
        if ACCESS_TOKEN not in metadata_dict:
            return False
        access_token_bytes = cast(bytes, metadata_dict[ACCESS_TOKEN])

        headers = {"Authorization": access_token_bytes.decode("utf-8")}

        response = post(self.validate_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True
        return False

    def get_auth_tokens(self, request: GetAuthTokensRequest) -> GetAuthTokensResponse:
        """Get the relevant auth tokens."""
        device_code = request.auth_details.get(DEVICE_CODE)
        if device_code is None:
            return GetAuthTokensResponse(auth_tokens={})

        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
            GRANT_TYPE: DEVICE_FLOW_GRANT_TYPE,
            DEVICE_CODE: device_code,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            access_token: str = data[ACCESS_TOKEN]
            refresh_token: str = data[REFRESH_TOKEN]
            auth_tokens = {
                ACCESS_TOKEN: access_token,
                REFRESH_TOKEN: refresh_token,
            }
            return GetAuthTokensResponse(auth_tokens=auth_tokens)

        return GetAuthTokensResponse(auth_tokens={})

    def refresh_tokens(self, context: grpc.ServicerContext) -> bool:
        """Refresh auth tokens in the metadata of the provided context."""
        metadata_dict = dict(context.invocation_metadata())
        if REFRESH_TOKEN not in metadata_dict:
            return False
        refresh_token_bytes: bytes = metadata_dict[REFRESH_TOKEN]

        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
            GRANT_TYPE: REFRESH_TOKEN,
            REFRESH_TOKEN: refresh_token_bytes.decode("utf-8"),
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:

            data: dict[str, Any] = response.json()
            new_access_token: str = data[ACCESS_TOKEN]
            new_refresh_token: str = data[REFRESH_TOKEN]

            metadata_sent = (
                (ACCESS_TOKEN, new_access_token),
                (REFRESH_TOKEN, new_refresh_token),
            )
            context.send_initial_metadata(metadata_sent)
            return True

        return False


class KeycloakCliPlugin(CliAuthPlugin):
    """Flower Keycloak Auth Plugin for CLI."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.access_token: str = config.get(ACCESS_TOKEN, "")
        self.refresh_token: str = config.get(REFRESH_TOKEN, "")
        self.config: dict[str, Any] = {}
        self.config_path = config_path

    @staticmethod
    def login(
        login_details: dict[str, str],
        config: dict[str, Any],
        federation: str,
        exec_stub: ExecStub,
    ) -> dict[str, Any]:
        """Login logic to log in user to the SuperLink."""
        timeout = int(login_details.get(EXPIRES_IN, "600"))
        interval = int(login_details.get(INTERVAL, "5"))
        device_code = login_details.get(DEVICE_CODE)
        verification_uri_complete = login_details.get(VERIFICATION_URI_COMPLETE)

        if device_code is None or verification_uri_complete is None:
            typer.secho(
                "❌ Missing information for Keycloak login .",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

        typer.secho(
            "Please login with your user credentials here: "
            f"{verification_uri_complete}",
            fg=typer.colors.BLUE,
        )
        start_time = time.time()
        time.sleep(interval)

        while (time.time() - start_time) < timeout:
            auth_details = {DEVICE_CODE: device_code}
            res: GetAuthTokensResponse = exec_stub.GetAuthTokens(
                GetAuthTokensRequest(auth_details=auth_details)
            )

            access_token = res.auth_tokens.get(ACCESS_TOKEN)
            refresh_token = res.auth_tokens.get(REFRESH_TOKEN)

            if access_token and refresh_token:
                config = {}
                config[AUTH_TYPE] = "keycloak"
                config[ACCESS_TOKEN] = access_token
                config[REFRESH_TOKEN] = refresh_token

                typer.secho(
                    "Login successful. You can now execute an authenticated "
                    "`flwr run` to the SuperLink.",
                    fg=typer.colors.GREEN,
                    bold=False,
                )
                return config

            time.sleep(interval)

        typer.secho(
            "❌ Timeout, failed to sign in.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> Sequence[tuple[str, Union[str, bytes]]]:
        """Write relevant auth tokens to the provided metadata."""
        return list(metadata) + [
            (ACCESS_TOKEN, self.access_token.encode("utf-8")),
            (REFRESH_TOKEN, self.refresh_token.encode("utf-8")),
        ]

    def store_tokens(self, config: dict[str, Any]) -> None:
        """Store the tokens from the provided config.

        The tokens will be saved as a JSON file in the `.credentials/` folder inside the
        Flower directory.
        """
        self.config = config
        self.access_token = config.get(ACCESS_TOKEN, "")
        self.refresh_token = config.get(REFRESH_TOKEN, "")

        with open(self.config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=4)

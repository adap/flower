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

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import grpc
import typer
from requests import post
from tomli_w import dump

from flwr.common.auth_plugin import CliAuthPlugin, ExecAuthPlugin
from flwr.common.constant import AUTH_TYPE
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokenRequest,
    GetAuthTokenResponse,
    LoginResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from .constant import (
    ACCESS_TOKEN,
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


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class KeycloakExecPlugin(ExecAuthPlugin):
    """Flower Keycloak Auth Plugin for ExecServicer."""

    def __init__(self, config: dict[str, Any]):
        self.auth_url = config.get(AUTH_URL, "")
        self.token_url = config.get(TOKEN_URL, "")
        self.keycloak_client_id = config.get(CLIENT_ID, "")
        self.keycloak_client_secret = config.get(CLIENT_SECRET, "")
        self.validate_url = config.get(VALIDATE_URL, "")

    def get_login_response(self) -> LoginResponse:
        """Send relevant auth url as a LoginResponse."""
        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.auth_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            device_code = data.get(DEVICE_CODE)
            verification_uri_complete = data.get(VERIFICATION_URI_COMPLETE)
            expires_in = data.get(EXPIRES_IN)
            interval = data.get(INTERVAL)
            login_details = {
                AUTH_TYPE: "keycloak",
                DEVICE_CODE: str(device_code),
                VERIFICATION_URI_COMPLETE: str(verification_uri_complete),
                EXPIRES_IN: str(expires_in),
                INTERVAL: str(interval),
            }
            return LoginResponse(login_details=login_details)

        return LoginResponse(login_details={})

    def validate_token_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> bool:
        """Authenticate auth tokens in the provided metadata."""
        access_token = _get_value_from_tuples(ACCESS_TOKEN, metadata)

        if not access_token:
            return False

        headers = {"Authorization": access_token.decode("utf-8")}

        response = post(self.validate_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True
        return False

    def get_auth_token_response(
        self, request: GetAuthTokenRequest
    ) -> GetAuthTokenResponse:
        """Send relevant tokens as a GetAuthTokenResponse."""
        device_code = request.auth_details.get(DEVICE_CODE)
        if device_code is None:
            return GetAuthTokenResponse(auth_tokens={})

        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
            GRANT_TYPE: DEVICE_FLOW_GRANT_TYPE,
            DEVICE_CODE: device_code,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            access_token = data.get(ACCESS_TOKEN)
            refresh_token = data.get(REFRESH_TOKEN)
            auth_tokens = {
                ACCESS_TOKEN: access_token,
                REFRESH_TOKEN: refresh_token,
            }
            return GetAuthTokenResponse(auth_tokens=auth_tokens)

        return GetAuthTokenResponse(auth_tokens={})

    def refresh_token(self, context: grpc.ServicerContext) -> bool:
        """Refresh auth tokens in the provided metadata."""
        metadata = context.invocation_metadata()
        refresh_token = _get_value_from_tuples(REFRESH_TOKEN, metadata)
        if not refresh_token:
            return False

        payload = {
            CLIENT_ID: self.keycloak_client_id,
            CLIENT_SECRET: self.keycloak_client_secret,
            GRANT_TYPE: REFRESH_TOKEN,
            REFRESH_TOKEN: refresh_token.decode("utf-8"),
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:

            data = response.json()
            access_token = data.get(ACCESS_TOKEN)
            refresh_token = data.get(REFRESH_TOKEN)

            context.send_initial_metadata(
                (
                    (
                        ACCESS_TOKEN,
                        access_token,
                    ),
                    (
                        REFRESH_TOKEN,
                        refresh_token,
                    ),
                )
            )
            return True

        return False


class KeycloakCliPlugin(CliAuthPlugin):
    """Flower Keycloak Auth Plugin for CLI."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.access_token = config[ACCESS_TOKEN]
        self.refresh_token = config[REFRESH_TOKEN]
        self.config = config
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
            res: GetAuthTokenResponse = exec_stub.GetAuthToken(
                GetAuthTokenRequest(auth_details=auth_details)
            )

            access_token = res.auth_tokens.get(ACCESS_TOKEN)
            refresh_token = res.auth_tokens.get(REFRESH_TOKEN)

            if access_token and refresh_token:
                config = {}
                config[AUTH_TYPE] = "keycloak"
                config[ACCESS_TOKEN] = access_token
                config[REFRESH_TOKEN] = refresh_token

                typer.secho(
                    "✅ Login successful.",
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

    def write_token_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> None:
        """Write relevant auth tokens to the provided metadata."""
        metadata.append(
            (
                ACCESS_TOKEN,
                self.access_token.encode("utf-8"),
            )
        )
        metadata.append(
            (
                REFRESH_TOKEN,
                self.refresh_token.encode("utf-8"),
            )
        )
        return metadata

    def store_refresh_token(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> None:
        """Store refresh tokens from the provided metadata.

        The tokens will be stored in the .credentials/ folder inside the Flower
        directory.
        """
        access_token = _get_value_from_tuples(ACCESS_TOKEN, metadata).decode("utf-8")
        refresh_token = _get_value_from_tuples(REFRESH_TOKEN, metadata).decode("utf-8")
        self.config[ACCESS_TOKEN] = access_token
        self.config[REFRESH_TOKEN] = refresh_token

        with open(self.config_path, "wb") as config_file:
            dump(self.config, config_file)

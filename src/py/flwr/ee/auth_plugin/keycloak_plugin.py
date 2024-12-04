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
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetAuthTokenRequest,
    GetAuthTokenResponse,
    LoginResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from .constant import (
    _ACCESS_TOKEN,
    _AUTH_TYPE,
    _AUTH_URL,
    _CLIENT_ID,
    _CLIENT_SECRET,
    _DEVICE_CODE,
    _DEVICE_FLOW_GRANT_TYPE,
    _EXPIRES_IN,
    _GRANT_TYPE,
    _INTERVAL,
    _REFRESH_TOKEN,
    _TOKEN_URL,
    _VALIDATE_URL,
    _VERIFICATION_URI_COMPLETE,
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
        self.auth_url = config.get(_AUTH_URL, "")
        self.token_url = config.get(_TOKEN_URL, "")
        self.keycloak_client_id = config.get(_CLIENT_ID, "")
        self.keycloak_client_secret = config.get(_CLIENT_SECRET, "")
        self.validate_url = config.get(_VALIDATE_URL, "")

    def get_login_response(self) -> LoginResponse:
        """Send relevant auth url as a LoginResponse."""
        payload = {
            _CLIENT_ID: self.keycloak_client_id,
            _CLIENT_SECRET: self.keycloak_client_secret,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.auth_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            device_code = data.get(_DEVICE_CODE)
            verification_uri_complete = data.get(_VERIFICATION_URI_COMPLETE)
            expires_in = data.get(_EXPIRES_IN)
            interval = data.get(_INTERVAL)
            login_details = {
                _AUTH_TYPE: "keycloak",
                _DEVICE_CODE: str(device_code),
                _VERIFICATION_URI_COMPLETE: str(verification_uri_complete),
                _EXPIRES_IN: str(expires_in),
                _INTERVAL: str(interval),
            }
            return LoginResponse(login_details=login_details)

        return LoginResponse(login_details={})

    def validate_token_in_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> bool:
        """Authenticate auth tokens in the provided metadata."""
        access_token = _get_value_from_tuples(_ACCESS_TOKEN, metadata)

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
        device_code = request.auth_details.get(_DEVICE_CODE)
        if device_code is None:
            return GetAuthTokenResponse(auth_tokens={})

        payload = {
            _CLIENT_ID: self.keycloak_client_id,
            _CLIENT_SECRET: self.keycloak_client_secret,
            _GRANT_TYPE: _DEVICE_FLOW_GRANT_TYPE,
            _DEVICE_CODE: device_code,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            access_token = data.get(_ACCESS_TOKEN)
            refresh_token = data.get(_REFRESH_TOKEN)
            auth_tokens = {
                _ACCESS_TOKEN: access_token,
                _REFRESH_TOKEN: refresh_token,
            }
            return GetAuthTokenResponse(auth_tokens=auth_tokens)

        return GetAuthTokenResponse(auth_tokens={})

    def refresh_token(self, context: grpc.ServicerContext) -> bool:
        """Refresh auth tokens in the provided metadata."""
        metadata = context.invocation_metadata()
        refresh_token = _get_value_from_tuples(_REFRESH_TOKEN, metadata)
        if not refresh_token:
            return False

        payload = {
            _CLIENT_ID: self.keycloak_client_id,
            _CLIENT_SECRET: self.keycloak_client_secret,
            _GRANT_TYPE: _REFRESH_TOKEN,
            _REFRESH_TOKEN: refresh_token.decode("utf-8"),
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:

            data = response.json()
            access_token = data.get(_ACCESS_TOKEN)
            refresh_token = data.get(_REFRESH_TOKEN)

            context.send_initial_metadata(
                (
                    (
                        _ACCESS_TOKEN,
                        access_token,
                    ),
                    (
                        _REFRESH_TOKEN,
                        refresh_token,
                    ),
                )
            )
            return True

        return False


class KeycloakCliPlugin(CliAuthPlugin):
    """Flower Keycloak Auth Plugin for CLI."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.access_token = config[_ACCESS_TOKEN]
        self.refresh_token = config[_REFRESH_TOKEN]
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
        timeout = int(login_details.get(_EXPIRES_IN, "600"))
        interval = int(login_details.get(_INTERVAL, "5"))
        device_code = login_details.get(_DEVICE_CODE)
        verification_uri_complete = login_details.get(_VERIFICATION_URI_COMPLETE)

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
            auth_details = {_DEVICE_CODE: device_code}
            res: GetAuthTokenResponse = exec_stub.GetAuthToken(
                GetAuthTokenRequest(auth_details=auth_details)
            )

            access_token = res.auth_tokens.get(_ACCESS_TOKEN)
            refresh_token = res.auth_tokens.get(_REFRESH_TOKEN)

            if access_token and refresh_token:
                config = {}
                config[_AUTH_TYPE] = "keycloak"
                config[_ACCESS_TOKEN] = access_token
                config[_REFRESH_TOKEN] = refresh_token

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

    def write_token_to_metadata(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> None:
        """Write relevant auth tokens to the provided metadata."""
        metadata.append(
            (
                _ACCESS_TOKEN,
                self.access_token.encode("utf-8"),
            )
        )
        metadata.append(
            (
                _REFRESH_TOKEN,
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
        access_token = _get_value_from_tuples(_ACCESS_TOKEN, metadata).decode("utf-8")
        refresh_token = _get_value_from_tuples(_REFRESH_TOKEN, metadata).decode("utf-8")
        self.config[_ACCESS_TOKEN] = access_token
        self.config[_REFRESH_TOKEN] = refresh_token

        with open(self.config_path, "wb") as config_file:
            dump(self.config, config_file)

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

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import grpc
import typer
from requests import get, post
from tomli_w import dump

from flwr.cli.utils import prompt_text
from flwr.common.auth_plugin import ExecAuthPlugin, Metadata, UserAuthPlugin
from flwr.proto.exec_pb2 import (
    GetAuthTokenRequest,
    GetAuthTokenResponse,
    LoginRequest,
    LoginResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

_AUTH_TOKEN_HEADER = "access-token"
_REFRESH_TOKEN_HEADER = "refresh-token"


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class KeycloakExecPlugin(ExecAuthPlugin):
    """Abstract Flower Exec API Auth Plugin class."""

    def __init__(self, config: Dict[str, Any]):
        self.auth_url = config.get(
            "auth_url",
            "https://idms-dev.flower.blue/realms/flwr-framework/protocol/openid-connect/auth/device",
        )
        self.token_url = config.get(
            "token_url",
            "https://idms-dev.flower.blue/realms/flwr-framework/protocol/openid-connect/token",
        )
        self.keycloak_client_id = config.get("client_id", "")
        self.keycloak_client_secret = config.get("client_secret", "")
        self.validate_url = config.get(
            "validate_url",
            "https://idms-dev.flower.blue/realms/flwr-framework/protocol/openid-connect/userinfo",
        )

    def send_auth_endpoint(self) -> LoginResponse:
        """Send relevant auth url as a LoginResponse."""
        payload = {
            "client_id": self.keycloak_client_id,
            "client_secret": self.keycloak_client_secret,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.auth_url, data=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            device_code = data.get("device_code")
            verification_uri_complete = data.get("verification_uri_complete")
            expires_in = data.get("expires_in")
            interval = data.get("interval")
            login_details = {
                "auth_type": "keycloak",
                "device_code": str(device_code),
                "verification_uri_complete": str(verification_uri_complete),
                "expires_in": str(expires_in),
                "interval": str(interval),
            }
            return LoginResponse(login_details=login_details)
        else:
            return LoginResponse(login_details={})

    def authenticate(self, metadata: Sequence[Tuple[str, Union[str, bytes]]]) -> bool:
        """Authenticate auth tokens in the provided metadata."""
        access_token = _get_value_from_tuples(_AUTH_TOKEN_HEADER, metadata)

        if not access_token:
            return False

        headers = {"Authorization": access_token.decode("utf-8")}

        response = post(self.validate_url, headers=headers)
        if response.status_code == 200:
            return True
        return False

    def get_auth_token_response(
        self, request: GetAuthTokenRequest
    ) -> GetAuthTokenResponse:
        """Send relevant tokens as a GetAuthTokenResponse."""
        device_code = request.auth_details.get("device_code")
        if device_code is None:
            return GetAuthTokenResponse(auth_tokens={})

        payload = {
            "client_id": self.keycloak_client_id,
            "client_secret": self.keycloak_client_secret,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            auth_tokens = {
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
            return GetAuthTokenResponse(auth_tokens=auth_tokens)
        else:
            return GetAuthTokenResponse(auth_tokens={})

    def refresh_token(self, context: grpc.ServicerContext) -> bool:
        """Refresh auth tokens in the provided metadata."""
        metadata = context.invocation_metadata()
        refresh_token = _get_value_from_tuples(_REFRESH_TOKEN_HEADER, metadata)
        if not refresh_token:
            return False

        payload = {
            "client_id": self.keycloak_client_id,
            "client_secret": self.keycloak_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token.decode("utf-8"),
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = post(self.token_url, data=payload, headers=headers)
        if response.status_code == 200:

            data = response.json()
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")

            context.send_initial_metadata(
                (
                    (
                        _AUTH_TOKEN_HEADER,
                        access_token,
                    ),
                    (
                        _REFRESH_TOKEN_HEADER,
                        refresh_token,
                    ),
                )
            )
            return True
        else:
            return False


class KeycloakUserPlugin(UserAuthPlugin):
    """Abstract Flower User Auth Plugin class."""

    def __init__(self, config: Dict[str, Any], config_path: Path):
        """Constructor for SuperTokensUserPlugin."""
        self.access_token = config["access-token"]
        self.refresh_token = config["refresh-token"]
        self.config = config
        self.config_path = config_path

    @staticmethod
    def login(
        login_details: Dict[str, str],
        config: Dict[str, Any],
        federation: str,
        exec_stub: ExecStub,
    ) -> Dict[str, Any]:
        """Read relevant auth details from federation config."""
        timeout = int(login_details.get("expires_in", "600"))
        interval = int(login_details.get("interval", "5"))
        device_code = login_details.get("device_code")
        verification_uri_complete = login_details.get("verification_uri_complete")

        if device_code is None or verification_uri_complete is None:
            typer.secho(
                "❌ Missing information for Keycloak login .",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

        typer.secho(
            f"Please login with your user credentials here: {verification_uri_complete}",
            fg=typer.colors.BLUE,
        )
        start_time = time.time()
        time.sleep(interval)

        while (time.time() - start_time) < timeout:
            auth_details = {"device_code": device_code}
            res: GetAuthTokenResponse = exec_stub.GetAuthToken(
                GetAuthTokenRequest(auth_details=auth_details)
            )

            access_token = res.auth_tokens.get("access_token")
            refresh_token = res.auth_tokens.get("refresh_token")

            if access_token and refresh_token:
                config = {}
                config["auth-type"] = "keycloak"
                config["access-token"] = access_token
                config["refresh-token"] = refresh_token

                typer.secho(
                    f"Login successful. You can now execute an authenticated `flwr run` to the SuperLink.",
                    fg=typer.colors.GREEN,
                    bold=False,
                )
                return config

            time.sleep(interval)

        typer.secho(
            f"❌ Timeout, failed to sign in.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    def provide_auth_details(self, metadata) -> Metadata:
        """Provide relevant auth tokens in the metadata."""
        metadata.append(
            (
                _AUTH_TOKEN_HEADER,
                self.access_token.encode("utf-8"),
            )
        )
        metadata.append(
            (
                _REFRESH_TOKEN_HEADER,
                self.refresh_token.encode("utf-8"),
            )
        )
        return metadata

    def save_refreshed_auth_tokens(
        self, metadata: Sequence[tuple[str, Union[str, bytes]]]
    ) -> None:
        """Provide refreshed auth tokens to the config file."""
        access_token = _get_value_from_tuples(_AUTH_TOKEN_HEADER, metadata).decode(
            "utf-8"
        )
        refresh_token = _get_value_from_tuples(_REFRESH_TOKEN_HEADER, metadata).decode(
            "utf-8"
        )
        self.config["access-token"] = access_token
        self.config["refresh-token"] = refresh_token

        with open(self.config_path, "wb") as config_file:
            dump(self.config, config_file)

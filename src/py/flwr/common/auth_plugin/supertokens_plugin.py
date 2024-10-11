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

from flwr.common.auth_plugin import SuperExecAuthPlugin, UserAuthPlugin, Metadata
from typing import Sequence, Tuple, Union, Any, List, Dict
from flwr.cli.utils import prompt_text
from requests import post, get
import typer
from tomli_w import dump
from flwr.proto.exec_pb2 import LoginRequest, LoginResponse

_AUTH_TOKEN_HEADER = "access-token"


def _get_value_from_tuples(
    key_string: str, tuples: Sequence[Tuple[str, Union[str, bytes]]]
) -> bytes:
    value = next((value for key, value in tuples if key == key_string), "")
    if isinstance(value, str):
        return value.encode()

    return value


class SuperTokensSuperExecPlugin(SuperExecAuthPlugin):
    """Abstract Flower SuperExec Auth Plugin class."""

    def __init__(self, config: Dict[str, Any]):
        self.auth_url = config.get("auth_url", "https://api.flower.ai/auth/signin")
        self.validate_url = config.get("validate_url", "https://api.flower.ai/api/v1/settings/profile/avatar")
            
    def send_auth_endpoint(self) -> LoginResponse:
        """
        Send relevant auth url as a LoginResponse.
        """
        return LoginResponse(auth_type="supertokens", auth_url=self.auth_url)

    def authenticate(self, metadata: Sequence[Tuple[str, Union[str, bytes]]]):
        """
        Authenticate auth tokens in the provided metadata.
        """
        access_token = _get_value_from_tuples(
            _AUTH_TOKEN_HEADER, metadata
        )
        url = "https://api.flower.ai/api/v1/settings/profile/avatar"
        cookies = {
            'sAccessToken': access_token.decode('utf-8')
        }
        response = get(url, cookies=cookies)
        if response.status_code == 200:
            return True
        return False


class SuperTokensUserPlugin(UserAuthPlugin):
    """Abstract Flower User Auth Plugin class."""

    def __init__(self, config: Dict[str, Any], federation: str):
        """Constructor for SuperTokensUserPlugin"""
        self.access_token = config["tool"]["flwr"]["federations"][federation]["authentication"]["access-token"]

    @staticmethod
    def login(auth_url: str, config: Dict[str, Any], federation: str) -> Dict[str, Any]:
        """
        Read relevant auth details from federation config.
        """
        email = prompt_text("Enter your email:")
        password = prompt_text("Enter your password:", hide_input=True)

        payload = {
            "formFields": [
                {"id": "email", "value": email},
                {"id": "password", "value": password},
            ]
        }
        headers = {"Content-Type": "application/json"}

        response = post(auth_url, json=payload, headers=headers)
        if response.status_code == 200:
            typer.secho("Sign-in successful", fg=typer.colors.GREEN)
            token = response.headers.get("st-access-token")

            if token:
                config["tool"]["flwr"]["federations"][federation]["authentication"] = {}
                config["tool"]["flwr"]["federations"][federation]["authentication"]["auth-type"] = "supertokens"
                config["tool"]["flwr"]["federations"][federation]["authentication"]["access-token"] = token
                return config
            else:
                typer.secho(
                    "❌ No access token found from response.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1)
        else:
            typer.secho(
                f"❌ Failed to sign in: {response.status_code}, "
                f"{response.text}",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

    def provide_auth_details(self, metadata) -> Metadata:
        """
        Provide relevant auth tokens in the metadata.
        """
        metadata.append(
            (
                _AUTH_TOKEN_HEADER,
                self.access_token.encode('utf-8'),
            )
        )
        return metadata

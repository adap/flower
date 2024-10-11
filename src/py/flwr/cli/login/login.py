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
"""Flower command line interface `login` command."""

import sys
from pathlib import Path
from flwr.common.logger import log
from logging import DEBUG
from typing import Annotated, Optional, Any

from flwr.cli.build import build
from flwr.cli.config_utils import load_and_validate
from flwr.common.config import flatten_dict, parse_config_args
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.proto.exec_pb2 import LoginRequest, LoginResponse  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub
from flwr.common.auth_plugin import SuperTokensUserPlugin, PublicKeyUserPlugin, UserAuthPlugin

import typer
from tomli_w import dump

from flwr.cli.config_utils import load_and_validate

auth_plugins = {
    "supertokens": SuperTokensUserPlugin,
    "public-key": PublicKeyUserPlugin
}

def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


def login(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to login into."),
    ] = None,
) -> None:
    """Login to Flower SuperExec."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / "pyproject.toml" if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)

    if config is None:
        typer.secho(
            "Project configuration could not be loaded.\n"
            "pyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        sys.exit()

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    typer.secho("Success", fg=typer.colors.GREEN)

    federation = federation or config["tool"]["flwr"]["federations"].get("default")

    if federation is None:
        typer.secho(
            "❌ No federation name was provided and the project's `pyproject.toml` "
            "doesn't declare a default federation (with a SuperExec address or an "
            "`options.num-supernodes` value).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Validate the federation exists in the configuration
    federation_config = config["tool"]["flwr"]["federations"].get(federation)
    if federation_config is None:
        available_feds = {
            fed for fed in config["tool"]["flwr"]["federations"] if fed != "default"
        }
        typer.secho(
            f"❌ There is no `{federation}` federation declared in "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)
    
    if "address" not in federation_config:
        typer.secho(
            f"❌ The federation `{federation}` does not have `SuperExec` "
            "address in its config.\n Please specify the address in "
            "`pyproject.toml` and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)
    
    login_response = send_login_request(app, federation_config)
    # login_response = LoginResponse(auth_type="supertokens", auth_url="https://api.flower.ai/auth/signin")
    auth_plugin = auth_plugins[login_response.auth_type]
    config = auth_plugin.login(login_response.auth_url, config, federation)

    with open(pyproject_path, "wb") as config_file:
        dump(config, config_file)

    
def send_login_request(app: Path, federation_config: dict[str, Any]) -> LoginResponse:
    insecure_str = federation_config.get("insecure")
    if root_certificates := federation_config.get("root-certificates"):
        root_certificates_bytes = (app / root_certificates).read_bytes()
        if insecure := bool(insecure_str):
            typer.secho(
                "❌ `root_certificates` were provided but the `insecure` parameter"
                "is set to `True`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
    else:
        root_certificates_bytes = None
        if insecure_str is None:
            typer.secho(
                "❌ To disable TLS, set `insecure = true` in `pyproject.toml`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
        if not (insecure := bool(insecure_str)):
            typer.secho(
                "❌ No certificate were given yet `insecure` is set to `False`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
    
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)
    stub = ExecStub(channel)

    req = LoginRequest()
    res: LoginResponse = stub.Login(req)

    return res
    
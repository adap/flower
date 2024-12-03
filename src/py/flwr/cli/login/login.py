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
from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from tomli_w import dump

from flwr.cli.config_utils import load_and_validate, validate_project_config, validate_federation_in_project_config
from flwr.common.config import get_flwr_dir
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import LoginRequest, LoginResponse  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub

try:
    from flwr.ee.auth_plugin import get_cli_auth_plugins
    auth_plugins = get_cli_auth_plugins()
except ImportError:
    auth_plugins = []


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


def login(  # pylint: disable=R0914
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

    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" not in federation_config:
        typer.secho(
            f"❌ The federation `{federation}` does not have `SuperExec` "
            "address in its config.\n Please specify the address in "
            "`pyproject.toml` and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    stub = _create_exec_stub(app, federation_config)
    login_request = LoginRequest()
    login_response: LoginResponse = stub.Login(login_request)
    auth_plugin = auth_plugins[login_response.login_details.get("auth_type", "")]
    config = auth_plugin.login(
        dict(login_response.login_details), config, federation, stub
    )

    base_path = get_flwr_dir()
    credentials_dir = base_path / ".credentials"
    credentials_dir.mkdir(parents=True, exist_ok=True)

    credential = credentials_dir / federation_config["address"]

    with open(credential, "wb") as config_file:
        dump(config, config_file)


def _create_exec_stub(app: Path, federation_config: dict[str, Any]) -> ExecStub:
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

    return stub

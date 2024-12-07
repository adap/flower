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

from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from flwr.cli.config_utils import (
    load_and_validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_project_config,
)
from flwr.common.config import get_user_auth_config_path
from flwr.common.constant import AUTH_TYPE
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from ..utils import try_obtain_cli_auth_plugin


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
    """Login to Flower SuperLink."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / "pyproject.toml" if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)

    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" not in federation_config:
        typer.secho(
            "❌ `flwr login` currently works with Exec API. Ensure that the correct"
            "Exec API address is provided in the `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    stub = _create_exec_stub(app, federation_config)
    login_request = GetLoginDetailsRequest()
    login_response: GetLoginDetailsResponse = stub.GetLoginDetails(login_request)

    # Get the auth plugin class and login
    auth_plugin_class = try_obtain_cli_auth_plugin(
        login_response.login_details.get(AUTH_TYPE, "")
    )
    config = auth_plugin_class.login(
        dict(login_response.login_details), config, federation, stub
    )

    # Store the tokens
    credential_path = get_user_auth_config_path(federation_config)
    auth_plugin = auth_plugin_class(config, credential_path)
    auth_plugin.store_tokens(config)


def _create_exec_stub(app: Path, federation_config: dict[str, Any]) -> ExecStub:
    insecure, root_certificates = validate_certificate_in_federation_config(
        app, federation_config
    )
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)
    stub = ExecStub(channel)

    return stub

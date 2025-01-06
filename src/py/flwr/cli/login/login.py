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


from pathlib import Path
from typing import Annotated, Optional

import typer

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.common.constant import AUTH_TYPE
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from ..utils import init_channel, try_obtain_cli_auth_plugin


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

    config = process_loaded_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )
    exit_if_no_address(federation_config, "login")
    channel = init_channel(app, federation_config, None)
    stub = ExecStub(channel)

    login_request = GetLoginDetailsRequest()
    login_response: GetLoginDetailsResponse = stub.GetLoginDetails(login_request)

    # Get the auth plugin
    auth_type = login_response.login_details.get(AUTH_TYPE)
    auth_plugin = try_obtain_cli_auth_plugin(app, federation, auth_type)
    if auth_plugin is None:
        typer.secho(
            f'❌ Authentication type "{auth_type}" not found',
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Login
    auth_config = auth_plugin.login(dict(login_response.login_details), stub)

    # Store the tokens
    auth_plugin.store_tokens(auth_config)

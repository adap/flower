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
"""Flower command line interface `login` command."""


from pathlib import Path
from typing import Annotated, Optional

import typer

from flwr.cli.auth_plugin import LoginError
from flwr.cli.config_utils import (
    exit_if_no_address,
    get_insecure_flag,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.typing import AccountAuthLoginDetails
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..utils import (
    account_auth_enabled,
    flwr_cli_grpc_exc_handler,
    init_channel,
    try_obtain_cli_auth_plugin,
)


def login(  # pylint: disable=R0914
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to login into."),
    ] = None,
    federation_config_overrides: Annotated[
        Optional[list[str]],
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
) -> None:
    """Login to Flower SuperLink."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / "pyproject.toml" if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)

    config = process_loaded_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config, federation_config_overrides
    )
    exit_if_no_address(federation_config, "login")

    # Check if `enable-user-auth` is set to `true`

    if not account_auth_enabled(federation_config):
        typer.secho(
            "❌ Account authentication is not enabled for the federation "
            f"'{federation}'. To enable it, set `enable-account-auth = true` "
            "in the federation configuration.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)
    # Check if insecure flag is set to `True`
    insecure = get_insecure_flag(federation_config)
    if insecure:
        typer.secho(
            "❌ `flwr login` requires TLS to be enabled. `insecure` must NOT be set to "
            "`true` in the federation configuration.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    channel = init_channel(app, federation_config, None)
    stub = ControlStub(channel)

    login_request = GetLoginDetailsRequest()
    with flwr_cli_grpc_exc_handler():
        login_response: GetLoginDetailsResponse = stub.GetLoginDetails(login_request)

    # Get the auth plugin
    auth_type = login_response.auth_type
    auth_plugin = try_obtain_cli_auth_plugin(
        app, federation, federation_config, auth_type
    )
    if auth_plugin is None:
        typer.secho(
            f'❌ Authentication type "{auth_type}" not found',
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Login
    details = AccountAuthLoginDetails(
        auth_type=login_response.auth_type,
        device_code=login_response.device_code,
        verification_uri_complete=login_response.verification_uri_complete,
        expires_in=login_response.expires_in,
        interval=login_response.interval,
    )
    try:
        with flwr_cli_grpc_exc_handler():
            credentials = auth_plugin.login(details, stub)
        typer.secho(
            "✅ Login successful.",
            fg=typer.colors.GREEN,
            bold=False,
        )
    except LoginError as e:
        typer.secho(
            f"❌ Login failed: {e.message}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from None

    # Store the tokens
    auth_plugin.store_tokens(credentials)

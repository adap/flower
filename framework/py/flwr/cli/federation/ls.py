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
"""Flower command line interface `federation list` command."""


import io
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    ListFederationsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611

from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def ls(  # pylint: disable=R0914, R0913, R0917
    ctx: typer.Context,
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """List SuperNodes in the federation."""
    # Resolve command used (list or ls)
    command_name = cast(str, ctx.command.name) if ctx.command else "ls"

    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    try:
        if suppress_output:
            redirect_output(captured_output)
        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        pyproject_path = app / FAB_CONFIG_FILE if app else None
        config, errors, warnings = load_and_validate(path=pyproject_path)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config
        )
        exit_if_no_address(federation_config, f"supernode {command_name}")
        channel = None
        try:
            auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)
            typer.echo("📄 Listing federations...")
            federations = _list_federations(stub)
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(data=_to_json(federations))
            else:
                Console().print(_to_table(federations))
        finally:
            if channel:
                channel.close()
    except (typer.Exit, Exception) as err:  # pylint: disable=broad-except
        if suppress_output:
            restore_output()
            e_message = captured_output.getvalue()
            print_json_error(e_message, err)
        else:
            typer.secho(
                f"{err}",
                fg=typer.colors.RED,
                bold=True,
            )
    finally:
        if suppress_output:
            restore_output()
        captured_output.close()


def _list_federations(stub: ControlStub) -> list[Federation]:
    """List all federations."""
    with flwr_cli_grpc_exc_handler():
        res: ListFederationsResponse = stub.ListFederations(ListFederationsRequest())

    return list(res.federations)


def _to_table(federations: list[Federation]) -> Table:
    """Format the provided federations list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Federation", justify="center"), style="bright_black", no_wrap=True
    )

    for federation in federations:
        table.add_row(federation.name)

    return table


def _to_json(federations: list[Federation]) -> list[dict[str, str]]:
    """Format the provided federations list to JSON serializable format."""
    return [{"name": federation.name} for federation in federations]

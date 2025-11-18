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
"""Flower command line interface `federation show` command."""


import io
from pathlib import Path
from typing import Annotated

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
from flwr.cli.ls import _get_status_style
from flwr.common.constant import FAB_CONFIG_FILE, NOOP_ACCOUNT_NAME, CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ShowFederationRequest,
    ShowFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611

from ..run_utils import RunRow, format_runs
from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def show(  # pylint: disable=R0914, R0913, R0917
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
    """Show details of a federation."""
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
        exit_if_no_address(federation_config, "federation show")
        federation_name = federation_config.get("federation", "")
        channel = None
        try:
            auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)
            typer.echo(f"📄 Showing '{federation_name}' federation ...")
            members, nodes, runs = _show_federation(stub, federation_name)
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(data=_to_json(members, nodes, runs))
            else:
                Console().print(_to_members_table(members))
                Console().print(_to_nodes_table(nodes))
                Console().print(_to_runs_table(runs))
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


def _show_federation(
    stub: ControlStub, federation: str
) -> tuple[list[str], list[NodeInfo], list[RunRow]]:
    """Show federation details."""
    with flwr_cli_grpc_exc_handler():
        res: ShowFederationResponse = stub.ShowFederation(
            ShowFederationRequest(federation_name=federation)
        )

    fed_proto = res.federation
    runs = [run_from_proto(run_proto) for run_proto in fed_proto.runs]
    formatted_runs = format_runs(runs, res.now)

    return fed_proto.member_aids, fed_proto.nodes, formatted_runs


def _to_members_table(members_aid: list[str]) -> Table:
    """Format the provided list of federation members as a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    table.add_column(
        Text("Member ID", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(Text("Role", justify="center"), style="bright_black", no_wrap=True)

    for member_aid in members_aid:
        table.add_row(member_aid, "Member")

    return table


def _to_nodes_table(nodes: list[NodeInfo]) -> Table:
    """Format the provided list of federation nodes as a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Node ID", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(Text("Owner", justify="center"))
    table.add_column(Text("Status", justify="center"))

    for row in nodes:
        owner_name = row.owner_name
        status = row.status

        if status == "online":
            status_style = "green"
        elif status == "offline":
            status_style = "bright_yellow"
        elif status == "unregistered":
            continue
        elif status == "registered":
            status_style = "blue"
        else:
            raise ValueError(f"Unexpected node status '{status}'")

        formatted_row = (
            f"[bold]{row.node_id}[/bold]",
            (
                f"{owner_name}"
                if owner_name != NOOP_ACCOUNT_NAME
                else f"[dim]{owner_name}[/dim]"
            ),
            f"[{status_style}]{status}",
        )
        table.add_row(*formatted_row)

    return table


def _to_runs_table(run_list: list[RunRow]) -> Table:
    """Format the provided list of federation runs as a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(Text("Run ID", justify="center"), no_wrap=True)
    table.add_column(Text("App", justify="center"))
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"), style="blue")

    for row in run_list:
        status_style = _get_status_style(row.status_text)

        formatted_row = (
            f"[bold]{row.run_id}[/bold]",
            f"@{row.fab_id}=={row.fab_version}",
            f"[{status_style}]{row.status_text}[/{status_style}]",
            row.elapsed,
        )
        table.add_row(*formatted_row)

    return table


def _to_json(
    members: list[str], nodes: list[NodeInfo], runs: list[RunRow]
) -> list[list[dict[str, str]]]:
    """Format the provided federation information to JSON serializable format."""
    members_list: list[dict[str, str]] = []
    nodes_list: list[dict[str, str]] = []
    runs_list: list[dict[str, str]] = []

    for member in members:
        members_list.append({"member_id": member, "role": "Member"})

    for node in nodes:
        nodes_list.append(
            {
                "node_id": node.node_id,
                "owner": node.owner_name,
                "status": node.status,
            }
        )

    for run in runs:
        runs_list.append(
            {
                "run_id": run.run_id,
                "app": f"@{run.fab_id}=={run.fab_version}",
                "status": run.status_text,
                "elapsed": run.elapsed,
            }
        )

    return [members_list, nodes_list, runs_list]

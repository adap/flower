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
"""Flower command line interface `supernode list` command."""


import io
import json
from datetime import datetime, timedelta
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
from flwr.common.constant import FAB_CONFIG_FILE, NOOP_ACCOUNT_NAME, CliOutputFormat
from flwr.common.date import format_timedelta, isoformat8601_utc
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListNodesRequest,
    ListNodesResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611

from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin

_NodeListType = tuple[int, str, str, str, str, str, str, str, str]


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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
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
            typer.echo("📄 Listing all nodes...")
            formatted_nodes = _list_nodes(stub)
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(_to_json(formatted_nodes, verbose=verbose))
            else:
                Console().print(_to_table(formatted_nodes, verbose=verbose))

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


def _list_nodes(stub: ControlStub) -> list[_NodeListType]:
    """List all nodes."""
    with flwr_cli_grpc_exc_handler():
        res: ListNodesResponse = stub.ListNodes(ListNodesRequest())

    return _format_nodes(list(res.nodes_info), res.now)


def _format_nodes(
    nodes_info: list[NodeInfo], now_isoformat: str
) -> list[_NodeListType]:
    """Format node information for display."""

    def _format_datetime(dt_str: str | None) -> str:
        dt = datetime.fromisoformat(dt_str) if dt_str else None
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    formatted_nodes: list[_NodeListType] = []
    # Add rows
    for node in sorted(
        nodes_info, key=lambda x: datetime.fromisoformat(x.registered_at)
    ):

        # Calculate elapsed times
        elapsed_time_activated = timedelta()
        if node.last_activated_at:
            end_time = datetime.fromisoformat(now_isoformat)
            elapsed_time_activated = end_time - datetime.fromisoformat(
                node.last_activated_at
            )

        formatted_nodes.append(
            (
                node.node_id,
                node.owner_aid,
                node.owner_name,
                node.status,
                _format_datetime(node.registered_at),
                _format_datetime(node.last_activated_at),
                _format_datetime(node.last_deactivated_at),
                _format_datetime(node.unregistered_at),
                format_timedelta(elapsed_time_activated),
            )
        )

    return formatted_nodes


def _to_table(nodes_info: list[_NodeListType], verbose: bool) -> Table:
    """Format the provided node list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Node ID", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(Text("Owner", justify="center"))
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"))
    table.add_column(Text("Status Changed @", justify="center"), style="bright_black")

    for row in nodes_info:
        (
            node_id,
            _,
            owner_name,
            status,
            _,
            last_activated_at,
            last_deactivated_at,
            unregistered_at,
            elapse_activated,
        ) = row

        if status == "online":
            status_style = "green"
            time_at = last_activated_at
        elif status == "offline":
            status_style = "bright_yellow"
            time_at = last_deactivated_at
        elif status == "unregistered":
            if not verbose:
                continue
            status_style = "red"
            time_at = unregistered_at
        elif status == "registered":
            status_style = "blue"
            time_at = "N/A"
        else:
            raise ValueError(f"Unexpected node status '{status}'")

        formatted_row = (
            f"[bold]{node_id}[/bold]",
            (
                f"{owner_name}"
                if owner_name != NOOP_ACCOUNT_NAME
                else f"[dim]{owner_name}[/dim]"
            ),
            f"[{status_style}]{status}",
            f"[cyan]{elapse_activated}[/cyan]" if status == "online" else "",
            time_at,
        )
        table.add_row(*formatted_row)

    return table


def _to_json(nodes_info: list[_NodeListType], verbose: bool) -> str:
    """Format node list to a JSON formatted string."""
    nodes_list = []
    for row in nodes_info:
        (
            node_id,
            owner_aid,
            owner_name,
            status,
            created_at,
            activated_at,
            deactivated_at,
            deleted_at,
            elapse_activated,
        ) = row

        if status == "deleted" and not verbose:
            continue

        nodes_list.append(
            {
                "node-id": node_id,
                "owner-aid": owner_aid,
                "owner-name": owner_name,
                "status": status,
                "created-at": created_at,
                "online-at": activated_at,
                "online-elapsed": elapse_activated,
                "offline-at": deactivated_at,
                "deleted-at": deleted_at,
            }
        )

    return json.dumps({"success": True, "nodes": nodes_list})

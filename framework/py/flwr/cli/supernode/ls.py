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
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

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
from flwr.common.date import isoformat8601_utc
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListNodesCliRequest,
    ListNodesCliResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611

from ..utils import flwr_cli_grpc_exc_handler, init_channel, try_obtain_cli_auth_plugin

_NodeListType = tuple[int, str, str, str, str, str]


def ls(  # pylint: disable=R0914
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
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
        exit_if_no_address(federation_config, "supernode list/ls")
        channel = None
        try:
            auth_plugin = try_obtain_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)
            typer.echo("📄 Listing all nodes...")
            formatted_nodes = _list_nodes(stub)
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(_to_json(formatted_nodes))
            else:
                Console().print(_to_table(formatted_nodes))

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
        res: ListNodesCliResponse = stub.ListNodesCli(ListNodesCliRequest())

    return _format_nodes(list(res.nodes_info), res.now)


def _format_nodes(
    nodes_info: list[NodeInfo], now_isoformat: str
) -> list[_NodeListType]:
    """Format node information for display."""

    def _format_datetime(dt: Optional[datetime]) -> str:
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    _ = now_isoformat
    formatted_nodes: list[_NodeListType] = []
    # Add rows
    for node in sorted(nodes_info, key=lambda x: datetime.fromisoformat(x.created_at)):

        # Convert isoformat to datetime
        created_at = (
            datetime.fromisoformat(node.created_at) if node.created_at else None
        )
        activated_at = (
            datetime.fromisoformat(node.activated_at) if node.activated_at else None
        )
        deactivated_at = (
            datetime.fromisoformat(node.deactivated_at) if node.deactivated_at else None
        )
        deleted_at = (
            datetime.fromisoformat(node.deleted_at) if node.deleted_at else None
        )

        formatted_nodes.append(
            (
                node.node_id,
                node.owner_aid,
                _format_datetime(created_at),
                _format_datetime(activated_at),
                _format_datetime(deactivated_at),
                _format_datetime(deleted_at),
            )
        )

    return formatted_nodes


def _to_table(nodes_info: list[_NodeListType]) -> Table:
    """Format the provided node list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Node ID", justify="center"), style="bright_white", overflow="fold"
    )
    table.add_column(Text("Owner", justify="center"), style="dim white")
    table.add_column(Text("Created At", justify="center"))
    table.add_column(Text("Activated At", justify="center"))
    table.add_column(Text("Deactivated At", justify="center"))
    table.add_column(Text("Deleted At", justify="center"))

    for row in nodes_info:
        (
            node_id,
            owner_aid,
            created_at,
            activated_at,
            deactivated_at,
            deleted_at,
        ) = row

        formatted_row = (
            f"[bold]{node_id}[/bold]",
            f"{owner_aid}",
            created_at,
            activated_at,
            deactivated_at,
            deleted_at,
        )
        table.add_row(*formatted_row)

    return table


def _to_json(nodes_info: list[_NodeListType]) -> str:
    """Format node list to a JSON formatted string."""
    nodes_list = []
    for row in nodes_info:
        (
            node_id,
            owner_aid,
            created_at,
            activated_at,
            deactivated_at,
            deleted_at,
        ) = row

        nodes_list.append(
            {
                "node-id": node_id,
                "owner-aid": owner_aid,
                "created-at": created_at,
                "activated-at": activated_at,
                "deactivated-at": deactivated_at,
                "deleted-at": deleted_at,
            }
        )

    return json.dumps({"success": True, "nodes": nodes_list})

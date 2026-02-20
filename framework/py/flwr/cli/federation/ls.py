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


from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.config_migration import migrate
from flwr.cli.flower_config import read_superlink_connection
from flwr.cli.ls import _get_status_style
from flwr.common.constant import NOOP_ACCOUNT_NAME, CliOutputFormat
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    ListFederationsResponse,
    ShowFederationRequest,
    ShowFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.utils import humanize_duration

from ..run_utils import RunRow, format_runs
from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)


def ls(  # pylint: disable=R0914, R0913, R0917, R0912
    ctx: typer.Context,
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
    federation: Annotated[
        str | None,
        typer.Option(
            "--federation",
            case_sensitive=False,
            help="Name of the federation to display",
        ),
    ] = None,
) -> None:
    """List available federations or details of a specific federation (alias: ls)."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Migrate legacy usage if any
        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)

            if federation:
                # Show specific federation details
                typer.echo(f"📄 Showing '{federation}' federation ...")
                members, nodes, runs = _show_federation(stub, federation)

                if is_json:
                    print_json_to_stdout(
                        _to_json(members=members, nodes=nodes, runs=runs)
                    )
                else:
                    Console().print(_to_members_table(members))
                    Console().print(_to_nodes_table(nodes))
                    Console().print(_to_runs_table(runs))
            else:
                # List federations
                typer.echo("📄 Listing federations...")
                federations = _list_federations(stub)

                if is_json:
                    print_json_to_stdout(_to_json(federations=federations))
                else:
                    Console().print(_to_table(federations))
        finally:
            if channel:
                channel.close()


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
    table.add_column(
        Text("Description", justify="center"), style="bright_black", no_wrap=True
    )

    for federation in federations:
        table.add_row(federation.name, federation.description)

    return table


def _to_json(
    federations: list[Federation] | None = None,
    members: list[str] | None = None,
    nodes: list[NodeInfo] | None = None,
    runs: list[RunRow] | None = None,
) -> list[dict[str, str]] | list[list[dict[str, Any]]]:
    """Format the provided federations list to JSON serializable format."""
    if federations is not None:
        return [{"name": federation.name} for federation in federations]

    if members is None or nodes is None or runs is None:
        return []

    members_list: list[dict[str, Any]] = []
    nodes_list: list[dict[str, Any]] = []
    runs_list: list[dict[str, Any]] = []

    for member in members:
        members_list.append({"member_id": member, "role": "Member"})

    for node in nodes:
        nodes_list.append(
            {
                "node_id": f"{node.node_id}",
                "owner": node.owner_name,
                "status": node.status,
            }
        )

    for run in runs:
        runs_list.append(
            {
                "run_id": f"{run.run_id}",
                "app": f"@{run.fab_id}=={run.fab_version}",
                "status": run.status_text,
                "elapsed": run.elapsed,
            }
        )

    return [members_list, nodes_list, runs_list]


def _show_federation(
    stub: ControlStub, federation: str
) -> tuple[list[str], list[NodeInfo], list[RunRow]]:
    """Show federation details.

    Parameters
    ----------
    stub : ControlStub
        The gRPC stub for Control API communication.
    federation : str
        Name of the federation to show.

    Returns
    -------
    tuple[list[str], list[NodeInfo], list[RunRow]]
        A tuple containing (account_names, nodes, runs).
    """
    with flwr_cli_grpc_exc_handler():
        res: ShowFederationResponse = stub.ShowFederation(
            ShowFederationRequest(federation_name=federation)
        )

    fed_proto = res.federation
    runs = [run_from_proto(run_proto) for run_proto in fed_proto.runs]
    formatted_runs = format_runs(runs, res.now)

    return (
        [account.name for account in fed_proto.accounts],
        list(fed_proto.nodes),
        formatted_runs,
    )


def _to_members_table(account_names: list[str]) -> Table:
    """Format the provided list of federation members as a rich Table.

    Parameters
    ----------
    account_names : list[str]
        List of member account names.

    Returns
    -------
    Table
        Rich Table object with formatted member information.
    """
    table = Table(title="Federation Members", header_style="bold cyan", show_lines=True)

    table.add_column(
        Text("Account Name", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(Text("Role", justify="center"), style="bright_black", no_wrap=True)

    for account_name in account_names:
        table.add_row(account_name, "Member")

    return table


def _to_nodes_table(nodes: list[NodeInfo]) -> Table:
    """Format the provided list of federation nodes as a rich Table.

    Parameters
    ----------
    nodes : list[NodeInfo]
        List of NodeInfo objects containing node details.

    Returns
    -------
    Table
        Rich Table object with formatted node information.

    Raises
    ------
    ValueError
        If an unexpected node status is encountered.
    """
    table = Table(
        title="SuperNodes in the Federation", header_style="bold cyan", show_lines=True
    )

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
    """Format the provided list of federation runs as a rich Table.

    Parameters
    ----------
    run_list : list[RunRow]
        List of RunRow objects containing run details.

    Returns
    -------
    Table
        Rich Table object with formatted run information.
    """
    table = Table(
        title="Runs in the Federation", header_style="bold cyan", show_lines=True
    )

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
            f"{humanize_duration(row.elapsed)}",
        )
        table.add_row(*formatted_row)

    return table

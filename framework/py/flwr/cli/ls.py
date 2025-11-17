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
"""Flower command line interface `ls` command."""


import io
import json
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
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat, Status, SubStatus
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from .run_utils import RunRow, format_runs
from .utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def ls(  # pylint: disable=too-many-locals, too-many-branches, R0913, R0917
    ctx: typer.Context,
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation"),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    runs: Annotated[
        bool,
        typer.Option(
            "--runs",
            help="List all runs",
        ),
    ] = False,
    run_id: Annotated[
        int | None,
        typer.Option(
            "--run-id",
            help="Specific run ID to display",
        ),
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
    """List the details of one provided run ID or all runs in a Flower federation.

    The following details are displayed:

    - **Run ID:** Unique identifier for the run.
    - **Federation:** The federation to which the run belongs.
    - **App:** The App associated with the run (``<APP_ID>==<APP_VERSION>``).
    - **Status:** Current status of the run (pending, starting, running, finished).
    - **Elapsed:** Time elapsed since the run started (``HH:MM:SS``).
    - **Status Changed @:** Timestamp of the most recent status change.

    All timestamps follow ISO 8601, UTC and are formatted as ``YYYY-MM-DD HH:MM:SSZ``.
    """
    # Resolve command used (list or ls)
    command_name = cast(str, ctx.command.name) if ctx.command else "list"

    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    try:
        if suppress_output:
            redirect_output(captured_output)
        # Load and validate federation config
        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        pyproject_path = app / FAB_CONFIG_FILE if app else None
        config, errors, warnings = load_and_validate(path=pyproject_path)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides
        )
        exit_if_no_address(federation_config, command_name)
        channel = None
        try:
            if runs and run_id is not None:
                raise ValueError(
                    "The options '--runs' and '--run-id' are mutually exclusive."
                )
            auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)

            # Display information about a specific run ID
            if run_id is not None:
                typer.echo(f"🔍 Displaying information for run ID {run_id}...")
                formatted_runs = _display_one_run(stub, run_id)
            # By default, list all runs
            else:
                typer.echo("📄 Listing all runs...")
                formatted_runs = _list_runs(stub)
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(_to_json(formatted_runs))
            else:
                if run_id is not None:
                    Console().print(_to_detail_table(formatted_runs[0]))
                else:
                    Console().print(_to_table(formatted_runs))
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


def _get_status_style(status_text: str) -> str:
    """Determine the display style/color for a status."""
    status = status_text.lower()
    sub_status = status_text.rsplit(":", maxsplit=1)[-1]

    if sub_status == SubStatus.COMPLETED:  # finished:completed
        return "green"
    if sub_status == SubStatus.FAILED:  # finished:failed
        return "red"
    if sub_status == SubStatus.STOPPED:  # finished:stopped
        return "yellow"
    if status in (Status.STARTING, Status.RUNNING):  # starting, running
        return "blue"
    return "bright_black"  # pending


def _to_table(run_list: list[RunRow]) -> Table:
    """Format the provided run list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(Text("Run ID", justify="center"), no_wrap=True)
    table.add_column(Text("Federation", justify="center"))
    table.add_column(Text("App", justify="center"))
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"), style="blue")
    table.add_column(Text("Status Changed @", justify="center"))

    for row in run_list:
        status_style = _get_status_style(row.status_text)

        # Use the most recent timestamp
        if row.finished_at != "N/A":
            status_changed_at = row.finished_at
        elif row.running_at != "N/A":
            status_changed_at = row.running_at
        elif row.starting_at != "N/A":
            status_changed_at = row.starting_at
        else:
            status_changed_at = row.pending_at

        formatted_row = (
            f"[bold]{row.run_id}[/bold]",
            row.federation,
            f"@{row.fab_id}=={row.fab_version}",
            f"[{status_style}]{row.status_text}[/{status_style}]",
            row.elapsed,
            status_changed_at,
        )
        table.add_row(*formatted_row)

    return table


def _to_detail_table(run: RunRow) -> Table:
    """Format a single run's details in a vertical table layout."""
    status_style = _get_status_style(run.status_text)

    # Create vertical table with field names on the left
    table = Table(show_header=False, show_lines=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    # Add rows with all details
    table.add_row("Run ID", f"[bold]{run.run_id}[/bold]")
    table.add_row("Federation", run.federation)
    table.add_row("App", f"@{run.fab_id}=={run.fab_version}")
    table.add_row("FAB Hash", f"{run.fab_hash[:8]}...{run.fab_hash[-8:]}")
    table.add_row("Status", f"[{status_style}]{run.status_text}[/{status_style}]")
    table.add_row("Elapsed", f"[blue]{run.elapsed}[/blue]")
    table.add_row("Pending At", run.pending_at)
    table.add_row("Starting At", run.starting_at)
    table.add_row("Running At", run.running_at)
    table.add_row("Finished At", run.finished_at)

    return table


def _to_json(run_list: list[RunRow]) -> str:
    """Format run status list to a JSON formatted string."""
    runs_list = []
    for row in run_list:
        runs_list.append(
            {
                "run-id": row.run_id,
                "federation": row.federation,
                "fab-id": row.fab_id,
                "fab-name": row.fab_id.split("/")[-1],
                "fab-version": row.fab_version,
                "fab-hash": row.fab_hash[:8],
                "status": row.status_text,
                "elapsed": row.elapsed,
                "pending-at": row.pending_at,
                "starting-at": row.starting_at,
                "running-at": row.running_at,
                "finished-at": row.finished_at,
            }
        )

    return json.dumps({"success": True, "runs": runs_list})


def _list_runs(stub: ControlStub) -> list[RunRow]:
    """List all runs."""
    with flwr_cli_grpc_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest())
    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    return format_runs(run_dict, res.now)


def _display_one_run(stub: ControlStub, run_id: int) -> list[RunRow]:
    """Display information about a specific run."""
    with flwr_cli_grpc_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest(run_id=run_id))
    if not res.run_dict:
        # This won't be reached as an gRPC error is raised if run_id is invalid
        raise ValueError(f"Run ID {run_id} not found")

    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    return format_runs(run_dict, res.now)

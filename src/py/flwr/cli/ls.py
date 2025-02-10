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
"""Flower command line interface `ls` command."""


import io
import json
from datetime import datetime, timedelta
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
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat, SubStatus
from flwr.common.date import format_timedelta, isoformat8601_utc
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.common.serde import run_from_proto
from flwr.common.typing import Run
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub

from .utils import init_channel, try_obtain_cli_auth_plugin, unauthenticated_exc_handler

_RunListType = tuple[int, str, str, str, str, str, str, str, str]


def ls(  # pylint: disable=too-many-locals, too-many-branches, R0913, R0917
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
    ] = None,
    federation_config_overrides: Annotated[
        Optional[list[str]],
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
        Optional[int],
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
    - **FAB:** Name of the FAB associated with the run (``{FAB_ID} (v{FAB_VERSION})``).
    - **Status:** Current status of the run (pending, starting, running, finished).
    - **Elapsed:** Time elapsed since the run started (``HH:MM:SS``).
    - **Created At:** Timestamp when the run was created.
    - **Running At:** Timestamp when the run started running.
    - **Finished At:** Timestamp when the run finished.

    All timestamps follow ISO 8601, UTC and are formatted as ``YYYY-MM-DD HH:MM:SSZ``.
    """
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
        exit_if_no_address(federation_config, "ls")
        channel = None
        try:
            if runs and run_id is not None:
                raise ValueError(
                    "The options '--runs' and '--run-id' are mutually exclusive."
                )
            auth_plugin = try_obtain_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ExecStub(channel)

            # Display information about a specific run ID
            if run_id is not None:
                typer.echo(f"🔍 Displaying information for run ID {run_id}...")
                restore_output()
                _display_one_run(stub, run_id, output_format)
            # By default, list all runs
            else:
                typer.echo("📄 Listing all runs...")
                restore_output()
                _list_runs(stub, output_format)

        except ValueError as err:
            if suppress_output:
                redirect_output(captured_output)
            typer.secho(
                f"❌ {err}",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from err
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


def _format_runs(run_dict: dict[int, Run], now_isoformat: str) -> list[_RunListType]:
    """Format runs to a list."""

    def _format_datetime(dt: Optional[datetime]) -> str:
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    run_list: list[_RunListType] = []

    # Add rows
    for run in sorted(
        run_dict.values(), key=lambda x: datetime.fromisoformat(x.pending_at)
    ):
        # Combine status and sub-status into a single string
        if run.status.sub_status == "":
            status_text = run.status.status
        else:
            status_text = f"{run.status.status}:{run.status.sub_status}"

        # Convert isoformat to datetime
        pending_at = datetime.fromisoformat(run.pending_at) if run.pending_at else None
        running_at = datetime.fromisoformat(run.running_at) if run.running_at else None
        finished_at = (
            datetime.fromisoformat(run.finished_at) if run.finished_at else None
        )

        # Calculate elapsed time
        elapsed_time = timedelta()
        if running_at:
            if finished_at:
                end_time = finished_at
            else:
                end_time = datetime.fromisoformat(now_isoformat)
            elapsed_time = end_time - running_at

        run_list.append(
            (
                run.run_id,
                run.fab_id,
                run.fab_version,
                run.fab_hash,
                status_text,
                format_timedelta(elapsed_time),
                _format_datetime(pending_at),
                _format_datetime(running_at),
                _format_datetime(finished_at),
            )
        )
    return run_list


def _to_table(run_list: list[_RunListType]) -> Table:
    """Format the provided run list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Run ID", justify="center"), style="bright_white", overflow="fold"
    )
    table.add_column(Text("FAB", justify="center"), style="dim white")
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"), style="blue")
    table.add_column(Text("Created At", justify="center"), style="dim white")
    table.add_column(Text("Running At", justify="center"), style="dim white")
    table.add_column(Text("Finished At", justify="center"), style="dim white")

    for row in run_list:
        (
            run_id,
            fab_id,
            fab_version,
            _,
            status_text,
            elapsed,
            created_at,
            running_at,
            finished_at,
        ) = row
        # Style the status based on its value
        sub_status = status_text.rsplit(":", maxsplit=1)[-1]
        if sub_status == SubStatus.COMPLETED:
            status_style = "green"
        elif sub_status == SubStatus.FAILED:
            status_style = "red"
        else:
            status_style = "yellow"

        formatted_row = (
            f"[bold]{run_id}[/bold]",
            f"{fab_id} (v{fab_version})",
            f"[{status_style}]{status_text}[/{status_style}]",
            elapsed,
            created_at,
            running_at,
            finished_at,
        )
        table.add_row(*formatted_row)

    return table


def _to_json(run_list: list[_RunListType]) -> str:
    """Format run status list to a JSON formatted string."""
    runs_list = []
    for row in run_list:
        (
            run_id,
            fab_id,
            fab_version,
            fab_hash,
            status_text,
            elapsed,
            created_at,
            running_at,
            finished_at,
        ) = row
        runs_list.append(
            {
                "run-id": run_id,
                "fab-id": fab_id,
                "fab-name": fab_id.split("/")[-1],
                "fab-version": fab_version,
                "fab-hash": fab_hash[:8],
                "status": status_text,
                "elapsed": elapsed,
                "created-at": created_at,
                "running-at": running_at,
                "finished-at": finished_at,
            }
        )

    return json.dumps({"success": True, "runs": runs_list})


def _list_runs(
    stub: ExecStub,
    output_format: str = CliOutputFormat.DEFAULT,
) -> None:
    """List all runs."""
    with unauthenticated_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest())
    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    formatted_runs = _format_runs(run_dict, res.now)
    if output_format == CliOutputFormat.JSON:
        Console().print_json(_to_json(formatted_runs))
    else:
        Console().print(_to_table(formatted_runs))


def _display_one_run(
    stub: ExecStub,
    run_id: int,
    output_format: str = CliOutputFormat.DEFAULT,
) -> None:
    """Display information about a specific run."""
    with unauthenticated_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest(run_id=run_id))
    if not res.run_dict:
        raise ValueError(f"Run ID {run_id} not found")

    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    formatted_runs = _format_runs(run_dict, res.now)
    if output_format == CliOutputFormat.JSON:
        Console().print_json(_to_json(formatted_runs))
    else:
        Console().print(_to_table(formatted_runs))

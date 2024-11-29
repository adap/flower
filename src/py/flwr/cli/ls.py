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


import json
import os
import re
import sys
from datetime import datetime, timedelta
from enum import Enum
from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import grpc
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.config_utils import (
    load_and_validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE, SubStatus
from flwr.common.date import format_timedelta, isoformat8601_utc
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log, restore_output
from flwr.common.serde import run_from_proto
from flwr.common.typing import Run
from flwr.proto.exec_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.proto.exec_pb2_grpc import ExecStub


class OutputFormat(str, Enum):
    """."""

    table = "table"  # pylint: disable=invalid-name
    json = "json"  # pylint: disable=invalid-name


def ls(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
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
    ls_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'table' or 'json'",
        ),
    ] = OutputFormat.table,
) -> None:
    """List runs."""
    suppress_output = ls_format == OutputFormat.json
    if suppress_output:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    # Load and validate federation config
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / FAB_CONFIG_FILE if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" not in federation_config:
        typer.secho(
            "❌ `flwr ls` currently works with Exec API. Ensure that the correct"
            "Exec API address is provided in the `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    try:
        if runs and run_id is not None:
            raise ValueError(
                "The options '--runs' and '--run-id' are mutually exclusive."
            )

        channel = _init_channel(app, federation_config)
        stub = ExecStub(channel)

        # Display information about a specific run ID
        if run_id is not None:
            typer.echo(f"🔍 Displaying information for run ID {run_id}...")
            restore_output()
            _display_one_run(
                stub, run_id, ls_format if ls_format is not None else "table"
            )
        # By default, list all runs
        else:
            typer.echo("📄 Listing all runs...")
            restore_output()
            _list_runs(stub, ls_format if ls_format is not None else "table")

    except ValueError as err:
        typer.secho(
            f"❌ {err}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err
    finally:
        channel.close()


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


def _init_channel(app: Path, federation_config: dict[str, Any]) -> grpc.Channel:
    """Initialize gRPC channel to the Exec API."""
    insecure, root_certificates_bytes = validate_certificate_in_federation_config(
        app, federation_config
    )
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)
    return channel


def _format_run(run_dict: dict[int, Run], now_isoformat: str) -> List[Tuple[str, ...]]:
    """Format run status as a rich Table."""

    def _format_datetime(dt: Optional[datetime]) -> str:
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    _list: List[Tuple[str, ...]] = []
    # Add rows
    for run in sorted(
        run_dict.values(), key=lambda x: datetime.fromisoformat(x.pending_at)
    ):
        # Combine status and sub-status into a single string
        if run.status.sub_status == "":
            status_text = run.status.status
        else:
            status_text = f"{run.status.status}:{run.status.sub_status}"

        # Style the status based on its value
        sub_status = run.status.sub_status
        if sub_status == SubStatus.COMPLETED:
            status_style = "green"
        elif sub_status == SubStatus.FAILED:
            status_style = "red"
        else:
            status_style = "yellow"

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

        _list.append(
            (
                f"[bold]{run.run_id}[/bold]",
                f"{run.fab_id} (v{run.fab_version})",
                f"[{status_style}]{status_text}[/{status_style}]",
                format_timedelta(elapsed_time),
                _format_datetime(pending_at),
                _format_datetime(running_at),
                _format_datetime(finished_at),
            )
        )
    return _list


def _to_table(run_list: List[Tuple[str, ...]]) -> Table:
    """."""
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
        table.add_row(*row)

    return table


def _to_json(run_list: List[Tuple[str, ...]]) -> str:
    """."""

    def _remove_bbcode_tags(strings: Tuple[str, ...]) -> Tuple[str, ...]:
        """Remove BBCode tags from the provided text."""
        # Regular expression pattern to match BBCode tags
        bbcode_pattern = re.compile(r"\[/?\w+\]")
        # Substitute BBCode tags with an empty string
        return tuple(bbcode_pattern.sub("", s) for s in strings)

    runs_dict: Dict[str, Dict[str, str]] = {}
    for row in run_list:
        row = _remove_bbcode_tags(row)
        run_id, fab, status, elapsed, created_at, running_at, finished_at = row
        runs_dict[run_id] = {
            "FAB": fab,
            "Status": status,
            "Elapsed": elapsed,
            "Created At": created_at,
            "Running At": running_at,
            "Finished At": finished_at,
        }
    return json.dumps(runs_dict)


def _list_runs(stub: ExecStub, format: str) -> None:
    """List all runs."""
    res: ListRunsResponse = stub.ListRuns(ListRunsRequest())
    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    if format == "table":
        Console().print(_to_table(_format_run(run_dict, res.now)))
    else:
        Console().print_json(_to_json(_format_run(run_dict, res.now)))


def _display_one_run(stub: ExecStub, run_id: int, format: str) -> None:
    """Display information about a specific run."""
    res: ListRunsResponse = stub.ListRuns(ListRunsRequest(run_id=run_id))
    if not res.run_dict:
        raise ValueError(f"Run ID {run_id} not found")

    run_dict = {run_id: run_from_proto(proto) for run_id, proto in res.run_dict.items()}

    if format == "table":
        Console().print(_to_table(_format_run(run_dict, res.now)))
    else:
        Console().print_json(_to_json(_format_run(run_dict, res.now)))

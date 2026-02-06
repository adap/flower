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
"""Flower command line interface `profile` command."""

import io
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.constant import CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import GetRunProfileRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub  # pylint: disable=E0611

from .utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def profile(
    run_id: Annotated[
        int,
        typer.Argument(help="The Flower run ID to query"),
    ],
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
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Get profiling summary for a run."""
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    try:
        if suppress_output:
            redirect_output(captured_output)

        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)
        pyproject_path = app / "pyproject.toml" if app else None
        config, errors, warnings = load_and_validate(pyproject_path, check_module=False)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides
        )
        exit_if_no_address(federation_config, "profile")

        auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
        channel = init_channel(app, federation_config, auth_plugin)
        try:
            stub = ControlStub(channel)
            req = GetRunProfileRequest(run_id=run_id)
            with flwr_cli_grpc_exc_handler():
                res = stub.GetRunProfile(req)
        finally:
            channel.close()

        if not res.summary_json:
            raise ValueError(f"No profile summary found for run ID {run_id}.")

        summary = json.loads(res.summary_json)
        restore_output()
        if output_format == CliOutputFormat.JSON:
            Console().print_json(json.dumps(summary))
        else:
            Console().print(_to_table(summary))
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
                err=True,
            )
    finally:
        if suppress_output:
            restore_output()
        captured_output.close()


def _to_table(summary: dict) -> Table:
    """Format the summary to a rich Table."""
    table = Table(title="Run Profile Summary")
    table.add_column("Task", style="white")
    table.add_column("Scope", style="bright_black")
    table.add_column("Round", style="cyan")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_column("Count", justify="right")

    entries = summary.get("entries", [])
    for entry in entries:
        table.add_row(
            str(entry.get("task", "")),
            str(entry.get("scope", "")),
            str(entry.get("round", "N/A")),
            f"{entry.get('avg_ms', 0.0):.2f}",
            f"{entry.get('max_ms', 0.0):.2f}",
            str(entry.get("count", 0)),
        )
    return table

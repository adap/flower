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
"""Flower command line interface `supernode unregister` command."""


import io
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import UnregisterNodeRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub

from ..utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def unregister(  # pylint: disable=R0914
    node_id: Annotated[
        int,
        typer.Argument(
            help="ID of the SuperNode to remove.",
        ),
    ],
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
    """Unregister a SuperNode from the federation."""
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
            federation, config
        )
        exit_if_no_address(federation_config, "supernode unregister")

        channel = None
        try:
            auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
            channel = init_channel(app, federation_config, auth_plugin)
            stub = ControlStub(channel)  # pylint: disable=unused-variable # noqa: F841

            _unregister_node(stub=stub, node_id=node_id, output_format=output_format)

        except ValueError as err:
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


def _unregister_node(
    stub: ControlStub,
    node_id: int,
    output_format: str,
) -> None:
    """Unregister a SuperNode from the federation."""
    with flwr_cli_grpc_exc_handler():
        stub.UnregisterNode(request=UnregisterNodeRequest(node_id=node_id))
    typer.secho(
        f"✅ SuperNode {node_id} unregistered successfully.", fg=typer.colors.GREEN
    )
    if output_format == CliOutputFormat.JSON:
        run_output = json.dumps(
            {
                "success": True,
                "node-id": node_id,
            }
        )
        restore_output()
        Console().print_json(run_output)

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
from typing import Annotated

import typer
from rich.console import Console

from flwr.cli.config_migration import migrate
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import UnregisterNodeRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub

from ..utils import flwr_cli_grpc_exc_handler, init_channel_from_connection


def unregister(  # pylint: disable=R0914
    node_id: Annotated[
        int,
        typer.Argument(
            help="ID of the SuperNode to remove.",
        ),
    ],
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
) -> None:
    """Unregister a SuperNode from the federation."""
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()

    if suppress_output:
        redirect_output(captured_output)

    # Migrate legacy usage if any
    migrate(superlink, args=[])

    # Read superlink connection configuration
    superlink_connection = read_superlink_connection(superlink)
    channel = None

    try:
        try:
            channel = init_channel_from_connection(
                superlink_connection, cmd="unregister"
            )
            stub = ControlStub(channel)

            _unregister_node(stub=stub, node_id=node_id, output_format=output_format)

        except ValueError as err:
            typer.secho(
                f"❌ {err}",
                fg=typer.colors.RED,
                bold=True,
                err=True,
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
                err=True,
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

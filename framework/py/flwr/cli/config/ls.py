# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `config list` command."""


import io
import json
from typing import Annotated

import typer
from rich.console import Console

from flwr.common.constant import CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output

from ..constant import SuperLinkConnectionTomlKey
from ..flower_config import read_flower_config


def ls(
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """List all SuperLink connections."""
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    config_path = None

    if suppress_output:
        redirect_output(captured_output)

    try:
        # Load Flower Config
        config, config_path = read_flower_config()

        # Get `superlink` tables
        superlink_connections = config.get(SuperLinkConnectionTomlKey.SUPERLINK, {})

        # Get default, then pop from dict
        default = superlink_connections.pop(SuperLinkConnectionTomlKey.DEFAULT, None)

        connection_names = list(superlink_connections.keys())
        restore_output()
        if output_format == CliOutputFormat.JSON:
            conn = {
                SuperLinkConnectionTomlKey.SUPERLINK: connection_names,
                SuperLinkConnectionTomlKey.DEFAULT: default,
            }
            Console().print_json(json.dumps(conn))
        else:
            typer.secho("Flower Config file: ", fg=typer.colors.BLUE, nl=False)
            typer.secho(f"{config_path}", fg=typer.colors.GREEN)
            typer.secho("SuperLink connections:", fg=typer.colors.BLUE)
            # List SuperLink connections and highlight default
            for k in connection_names:
                typer.secho(f"  {k}", fg=typer.colors.GREEN, nl=False)
                if k == default:
                    typer.secho(
                        f" ({SuperLinkConnectionTomlKey.DEFAULT})",
                        fg=typer.colors.WHITE,
                        nl=False,
                    )
                typer.echo()
            typer.echo()
    except typer.Exit as err:
        # log the error if json format requested
        # else do nothing since it will be logged by typer
        if suppress_output:
            restore_output()
            e_message = captured_output.getvalue()
            print_json_error(e_message, err)

    except Exception as err:  # pylint: disable=broad-except
        if suppress_output:
            restore_output()
            e_message = captured_output.getvalue()
            print_json_error(e_message, err)
        else:
            typer.secho(
                f"❌ An unexpected error occurred while listing the SuperLink "
                f"connections in the Flower configuration file ({config_path}): {err}",
                fg=typer.colors.RED,
                err=True,
            )

    finally:
        if suppress_output:
            restore_output()
        captured_output.close()

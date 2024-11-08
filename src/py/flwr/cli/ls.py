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


from typing import Optional, Annotated
from pathlib import Path
import typer


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
) -> None:
    """List runs."""
    try:
        if runs and run_id is not None:
            raise ValueError("The options '--runs' and '--run-id' are mutually exclusive.")

        if runs:
            # Logic to list all runs
            typer.echo(f"📄 Listing all runs for app at {app} with federation '{federation}'")
            # [Your code to list runs goes here]

        elif run_id is not None:
            # Logic to display information about a specific run ID
            typer.echo(f"🔍 Displaying information for run ID {run_id} in app at {app} with federation '{federation}'")
            # [Your code to display run information goes here]

        else:
            raise ValueError("You must specify either '--runs' or '--run-id'.")

    except ValueError as err:
        typer.secho(
            f"❌ {err}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err


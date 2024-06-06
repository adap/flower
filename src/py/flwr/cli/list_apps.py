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
"""Flower command line interface `build` command."""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .config_utils import load_and_validate


# pylint: disable=too-many-locals
def list_apps(
    versions: Annotated[
        bool,
        typer.Option(help="Whether or not to display the available versions."),
    ] = False,
    flwr_dir: Annotated[
        Optional[Path],
        typer.Option(help="The Flower directory, `$HOME/.flwr/` by default."),
    ] = None,
) -> None:
    """List all the installed Flower Apps."""
    if flwr_dir is None:
        flwr_dir = Path(Path.home() / ".flwr")

    typer.secho(
        f"Here is the list of valid Flower Apps found under `{flwr_dir}/apps/` :\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    for username_dir in (flwr_dir / "apps").iterdir():
        for app_dir in username_dir.iterdir():
            conf = None
            v_list = []
            if versions:
                for v_app in app_dir.iterdir():
                    conf, _, _ = load_and_validate(v_app / "pyproject.toml")
                    v_list.append(v_app.name)
            else:
                conf, _, _ = load_and_validate(
                    next(app_dir.iterdir()) / "pyproject.toml"
                )

            if conf is None:
                continue
            else:
                typer.secho(
                    f"\t * {username_dir.name}/{app_dir.name}"
                    f"{' (' + ', '.join(v_list) + ')' if versions else ''}",
                    fg=typer.colors.GREEN,
                    bold=True,
                )

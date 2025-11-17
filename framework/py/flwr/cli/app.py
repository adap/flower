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
"""Flower command line interface."""

import typer
from typer.main import get_command

from flwr.common.version import package_version

from .build import build
from .federation import ls as federation_list
from .install import install
from .log import log
from .login import login
from .ls import ls
from .new import new
from .pull import pull
from .run import run
from .stop import stop
from .supernode import ls as supernode_list
from .supernode import register as supernode_register
from .supernode import unregister as supernode_unregister

app = typer.Typer(
    help=typer.style(
        "flwr is the Flower command line interface.",
        fg=typer.colors.BRIGHT_YELLOW,
        bold=True,
    ),
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command()(new)
app.command()(run)
app.command()(build)
app.command()(install)
app.command()(log)
app.command("list")(ls)
app.command(hidden=True)(ls)
app.command()(stop)
app.command()(login)
app.command()(pull)

# Create supernode command group
supernode_app = typer.Typer(help="Manage SuperNodes")
supernode_app.command()(supernode_register)
supernode_app.command()(supernode_unregister)
# Make it appear as "list"
supernode_app.command("list")(supernode_list)
# Hide "ls" command (left as alias)
supernode_app.command(hidden=True)(supernode_list)
app.add_typer(supernode_app, name="supernode")

# Create federation command group
federation_app = typer.Typer(help="Manage Federations")
# Make it appear as "list"
federation_app.command("list")(federation_list)
# Hide "ls" command (left as alias)
federation_app.command(hidden=True)(federation_list)
app.add_typer(federation_app, name="federation")

typer_click_object = get_command(app)


@app.callback(invoke_without_command=True)
def version_callback(
    ver: bool = typer.Option(
        None,
        "-V",
        "--version",
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Print version."""
    if ver:
        typer.secho(f"Flower version: {package_version}", fg="blue")
        raise typer.Exit()


if __name__ == "__main__":
    app()

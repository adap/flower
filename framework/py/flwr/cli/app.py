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


from typing import Any, TypedDict

import typer
from typer.main import get_command

from flwr.supercore.version import package_version

from .app_cmd import publish as app_publish
from .app_cmd import review as app_review
from .build import build
from .config import ls as config_list
from .federation import add_supernode as federation_add_supernode
from .federation import archive as federation_archive
from .federation import create as federation_create
from .federation import ls as federation_list
from .federation import remove_supernode as federation_remove_supernode
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


class CommandKwargs(TypedDict):
    """Keywords for typer command to make mypy happy."""

    context_settings: dict[str, Any]


ALLOW_EXTRAS: CommandKwargs = {"context_settings": {"allow_extra_args": True}}

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
app.command(**ALLOW_EXTRAS)(log)
app.command("list", **ALLOW_EXTRAS)(ls)
app.command(hidden=True, **ALLOW_EXTRAS)(ls)
app.command(**ALLOW_EXTRAS)(stop)
app.command(**ALLOW_EXTRAS)(login)
app.command(**ALLOW_EXTRAS)(pull)

# Create supernode command group
supernode_app = typer.Typer(help="Manage SuperNodes")
supernode_app.command(**ALLOW_EXTRAS)(supernode_register)
supernode_app.command(**ALLOW_EXTRAS)(supernode_unregister)
# Make it appear as "list"
supernode_app.command("list", **ALLOW_EXTRAS)(supernode_list)
# Hide "ls" command (left as alias)
supernode_app.command(hidden=True, **ALLOW_EXTRAS)(supernode_list)
app.add_typer(supernode_app, name="supernode")

# Create app command group
app_app = typer.Typer(help="Manage Apps")
app_app.command()(app_review)
app_app.command()(app_publish)
app.add_typer(app_app, name="app")

# Create federation command group
federation_app = typer.Typer(help="Manage Federations")
# Make it appear as "list"
federation_app.command("list", **ALLOW_EXTRAS)(federation_list)
# Hide "ls" command (left as alias)
federation_app.command(hidden=True, **ALLOW_EXTRAS)(federation_list)
federation_app.command(**ALLOW_EXTRAS)(federation_archive)
federation_app.command(**ALLOW_EXTRAS)(federation_create)
federation_app.command("add-supernode", **ALLOW_EXTRAS)(federation_add_supernode)
federation_app.command("remove-supernode", **ALLOW_EXTRAS)(federation_remove_supernode)
app.add_typer(federation_app, name="federation")

# Create config command group
config_app = typer.Typer(help="Manage Configuration")
config_app.command("list")(config_list)
# Hide "ls" command (left as alias)
config_app.command(hidden=True)(config_list)
app.add_typer(config_app, name="config")

typer_click_object = get_command(app)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        None,
        "-V",
        "--version",
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Flower CLI."""
    if version:
        typer.secho(f"Flower version: {package_version}", fg="blue")
        raise typer.Exit()


if __name__ == "__main__":
    app()

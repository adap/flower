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
"""Flower command line interface `connect` command."""


from pathlib import Path
from typing import Annotated, Optional

import typer

from flwr.cli.config_utils import load_and_validate, process_loaded_project_config

from .utils import is_valid_project_name, prompt_text, sanitize_project_name


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def connect(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project to run"),
    ] = Path("."),
    federation_name: Annotated[
        Optional[str],
        typer.Argument(help="The name of the federation"),
    ] = None,
    superlink_address: Annotated[
        Optional[str],
        typer.Option(case_sensitive=False, help="The address of the SuperLink to use"),
    ] = None,
    insecure: Annotated[
        Optional[bool],
        typer.Option(case_sensitive=False, help="Wether the connection is insecure."),
    ] = None,
    cert_path: Annotated[
        Optional[str],
        typer.Option(
            case_sensitive=False, help="The path of certificates to use for SSL."
        ),
    ] = None,
) -> None:
    """Create new Flower App."""
    pyproject_path = app / "pyproject.toml"
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = process_loaded_project_config(config, errors, warnings)

    if federation_name is None:
        federation_name = prompt_text("Please provide the federation name to add")
    if not is_valid_project_name(federation_name):
        federation_name = prompt_text(
            "Please provide a name that only contains "
            "characters in {'-', a-zA-Z', '0-9'}",
            predicate=is_valid_project_name,
            default=sanitize_project_name(federation_name),
        )

    if config["tool"]["flwr"]["federations"].get(federation_name):
        federation_name = prompt_text(
            "Please provide a name that only contains "
            "characters in {'-', a-zA-Z', '0-9'}",
            predicate=lambda fed_name: not config["tool"]["flwr"]["federations"].get(
                fed_name
            ),
            default=f"{federation_name}-1",
        )

    if superlink_address is None:
        superlink_address = prompt_text(
            "Please provide the address of the SuperLink you wish to connect to"
        )

    if insecure is None and not cert_path:
        insecure = not typer.confirm(
            typer.style(
                "\n💬 Do you want to use a secure SSL connection?",
                fg=typer.colors.MAGENTA,
                bold=True,
            ),
            default=False,
        )
    if not insecure and not cert_path:
        cert_path = prompt_text(
            "Please provide the path of the SSL certifications",
        )
        cert_path = cert_path.lower()

    print(
        typer.style(
            f"\n🔨 Creating Flower App {federation_name}...",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    new_content = (
        f"{pyproject_path.read_text()}\n"
        f"[tool.flwr.federations.{federation_name}]\n"
        f'address = "{superlink_address}"\n'
    )

    if cert_path:
        new_content += f'root-certificates = "{cert_path}"\n'
    elif insecure:
        new_content += "insecure = true\n"
    else:
        typer.secho(
            "No root certificates were provided and insecure was set to false",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from None

    pyproject_path.write_text(new_content)

    prompt = typer.style(
        f"🎊 {app} successfully connected to {superlink_address}.\n\n"
        f"To run your Flower App using {federation_name}:\n\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        f"	cd {app} && pip install -e .\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    prompt += typer.style(
        "then, run the app:\n\n ",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        f"\tflwr run . {federation_name}\n\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    print(prompt)

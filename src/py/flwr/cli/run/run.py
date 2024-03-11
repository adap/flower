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
"""Flower command line interface `run` command."""

import sys

import typer

from flwr.cli.flower_toml import apply_defaults, load_flower_toml, validate_flower_toml
from flwr.simulation.run_simulation import _run_simulation


def run() -> None:
    """Run Flower project."""
    print(
        typer.style("Loading project configuration... ", fg=typer.colors.BLUE),
        end="",
    )
    config = load_flower_toml()
    if not config:
        print(
            typer.style(
                "Project configuration could not be loaded. "
                "flower.toml does not exist.",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        sys.exit()
    print(typer.style("Success", fg=typer.colors.GREEN))

    print(
        typer.style("Validating project configuration... ", fg=typer.colors.BLUE),
        end="",
    )
    is_valid, errors, warnings = validate_flower_toml(config)
    if warnings:
        print(
            typer.style(
                "Project configuration is missing the following "
                "recommended properties:\n"
                + "\n".join([f"- {line}" for line in warnings]),
                fg=typer.colors.RED,
                bold=True,
            )
        )

    if not is_valid:
        print(
            typer.style(
                "Project configuration could not be loaded.\nflower.toml is invalid:\n"
                + "\n".join([f"- {line}" for line in errors]),
                fg=typer.colors.RED,
                bold=True,
            )
        )
        sys.exit()
    print(typer.style("Success", fg=typer.colors.GREEN))

    # Apply defaults
    defaults = {
        "flower": {
            "engine": {"name": "simulation", "simulation": {"supernode": {"num": 2}}}
        }
    }
    config = apply_defaults(config, defaults)

    server_app_ref = config["flower"]["components"]["serverapp"]
    client_app_ref = config["flower"]["components"]["clientapp"]
    engine = config["flower"]["engine"]["name"]

    if engine == "simulation":
        num_supernodes = config["flower"]["engine"]["simulation"]["supernode"]["num"]

        print(
            typer.style("Starting run... ", fg=typer.colors.BLUE),
        )
        _run_simulation(
            server_app_attr=server_app_ref,
            client_app_attr=client_app_ref,
            num_supernodes=num_supernodes,
        )
    else:
        print(
            typer.style(
                f"Engine '{engine}' is not yet supported in `flwr run`",
                fg=typer.colors.RED,
                bold=True,
            )
        )

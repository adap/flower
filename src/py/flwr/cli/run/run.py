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
"""Flower command line interface `new` command."""

import typer

from flwr.cli.flower_toml import load_flower_toml, validate_flower_toml


def run() -> None:
    """Run Flower project."""
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

    is_valid, reasons = validate_flower_toml(config)

    if not is_valid:
        print(
            typer.style(
                "Project configuration could not be loaded.\nflower.toml is invalid:\n"
                + "\n".join([f"- {line}" for line in reasons]),
                fg=typer.colors.RED,
                bold=True,
            )
        )

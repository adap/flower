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
"""Flower command line interface `supernode add` command."""

from pathlib import Path
from typing import Annotated, Optional

import typer


def add(  # pylint: disable=R0914
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
    ] = None,
    pub_key: Annotated[
        Path,
        typer.Option(
            "--pub-key",
            help="Path to the public key file.",
        ),
    ] = None,
) -> None:
    """Add a SuperNode to the federation."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

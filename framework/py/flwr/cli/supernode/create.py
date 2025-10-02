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
"""Flower command line interface `supernode create` command."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE


def create(  # pylint: disable=R0914
    public_key: Annotated[
        Path,
        typer.Argument(
            help="Path to the public key file.",
        ),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
    ] = None,
) -> None:
    """Add a SuperNode to the federation."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / FAB_CONFIG_FILE if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = process_loaded_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )
    exit_if_no_address(federation_config, "supernode add")

    _ = public_key

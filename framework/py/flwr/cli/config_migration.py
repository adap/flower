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
"""Utilities for migrating old TOML configurations to Flower config."""


import re
from pathlib import Path
from typing import Any

import click
import typer

from .config_utils import load_and_validate, validate_federation_in_project_config
from .flower_config import (
    init_flwr_config,
    parse_superlink_connection,
    set_default_superlink_connection,
    write_superlink_connection,
)

CONFIG_MIGRATION_NOTICE = """
##################################################################
# CONFIGURATION MIGRATION NOTICE:
#
# What was previously called "federation config" in pyproject.toml
# has been renamed and moved.
#
# Federation config is now **SuperLink connection configuration**
# and is defined in the Flower configuration file.
#
# The entries below are commented out intentionally and are kept
# only as a migration reference.
#
# Docs: <link to Flower config docs>
##################################################################

"""


def _is_legacy_usage(superlink: str, args: list[str]) -> bool:
    """Check if legacy usage is detected in the given arguments."""
    # If one and only one extra argument is given, assume legacy usage
    if len(args) == 1:
        return True

    # If the `superlink` looks like a path, assume legacy usage
    pth = Path(superlink)
    if pth.is_absolute() or len(pth.parts) > 1 or superlink in (".", ".."):
        return True

    # Lastly, check if a pyproject.toml file exists at the given superlink
    if (pth / "pyproject.toml").exists():
        return True

    return False


def _migrate_pyproject_toml_to_flower_config(
    app: Path, toml_federation: str | None
) -> None:
    """Migrate old TOML configuration to Flower config."""
    # Load and validate the old TOML configuration
    toml_path = app / "pyproject.toml"
    config, _, _ = load_and_validate(toml_path, check_module=False)
    if config is None:
        raise ValueError(f"Failed to load TOML configuration: {toml_path}")
    validate_federation_in_project_config(toml_federation, config)

    # Construct SuperLinkConnection
    toml_federations: dict[str, Any] = config["tool"]["flwr"]["federations"]
    for name, toml_fed_config in toml_federations.items():
        if isinstance(toml_fed_config, dict):
            conn = parse_superlink_connection(toml_fed_config, name)
            write_superlink_connection(conn)

    # Set default federation if applicable
    default_toml_federation: str | None = toml_federations.get("default")
    if default_toml_federation in toml_federations:
        set_default_superlink_connection(default_toml_federation)


def _comment_out_legacy_toml_config(app: Path) -> None:
    """Comment out legacy TOML configuration in pyproject.toml."""
    # Read pyproject.toml lines
    toml_path = app / "pyproject.toml"
    lines = toml_path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_pattern = re.compile(r"\s*\[(.*)\]")

    # Comment out the [tool.flwr.federations] section
    notice_added = False
    in_federation_section = False
    with toml_path.open("w", encoding="utf-8") as f:
        for line in lines:
            # Detect section headers
            if match := section_pattern.match(line):
                section = match.group(1)
                in_federation_section = section.startswith("tool.flwr.federations")

            # Comment out lines in the federation section
            if in_federation_section:
                if not notice_added:
                    f.write(CONFIG_MIGRATION_NOTICE)
                    notice_added = True
                f.write(f"# {line}")
            else:
                f.write(line)


def migrate_if_legacy_usage(
    superlink: str,
    args: list[str],
) -> None:
    """Migrate legacy TOML configuration to Flower config if legacy usage is
    detected."""
    # Initialize Flower config
    init_flwr_config()

    # Trigger the same typer error when detecting unexpected extra args
    if len(args) > 1:
        raise click.UsageError(f"Got unexpected extra arguments ({' '.join(args[1:])})")

    # Skip migration if no legacy usage is detected
    if not _is_legacy_usage(superlink, args):
        return

    # Check if pyproject.toml exists
    app_path = Path(superlink)
    if not (app_path / "pyproject.toml").exists():
        raise click.ClickException(
            "Legacy usage detected, but no pyproject.toml found "
            f"at '{app_path.absolute()}'."
        )
    
    # Prompt user for confirmation
    confirm = typer.confirm(
        typer.style(
            f"\n💬 Legacy TOML configuration detected at '{app_path.absolute()}'. "
            "Do you want to migrate it to Flower config?",
            fg=typer.colors.MAGENTA,
            bold=True,
        ),
        default=True,
    )
    if not confirm:
        raise click.ClickException("Migration aborted by user.")

    try:
        _migrate_pyproject_toml_to_flower_config(
            app=app_path,
            toml_federation=args[0] if args else None,
        )
    except Exception as e:
        raise click.ClickException(
            "Failed to migrate legacy TOML configuration to Flower config."
        ) from e

    _comment_out_legacy_toml_config(app_path)

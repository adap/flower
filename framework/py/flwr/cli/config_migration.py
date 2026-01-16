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
# What was previously called "federation config" for SuperLink
# connections in pyproject.toml has been renamed and moved.
#
# These settings are now **SuperLink connection configuration**
# and are defined in the Flower configuration file.
#
# The entries below are commented out intentionally and are kept
# only as a migration reference.
#
# Docs: <link to Flower config docs>
##################################################################

"""

CLI_NOTICE = (
    typer.style("\n🌸 Heads up from Flower!\n\n", fg=typer.colors.MAGENTA, bold=True)
    + "We detected legacy usage of this command that relies on connection\n"
    + "settings from your pyproject.toml.\n\n"
    + "Flower will migrate any relevant settings to the new Flower config.\n\n"
    + "Learn more: https://flower.ai/docs\n"
)


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


def _check_is_migratable(app: Path) -> None:
    """Check if the given app path contains legacy TOML configuration."""
    toml_path = app / "pyproject.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"No pyproject.toml found in '{app}'")
    config, errors, _ = load_and_validate(toml_path, check_module=False)
    if config is None:
        raise ValueError(f"Failed to load TOML configuration: {toml_path}")
    if errors:
        raise ValueError(
            f"Invalid TOML configuration found in '{toml_path}':\n"
            + "\n".join(f"- {err}" for err in errors)
        )
    try:
        _ = config["tool"]["flwr"]["federations"]
        return
    except KeyError:
        raise ValueError(
            f"No 'tool.flwr.federations' section found in '{toml_path}'"
        ) from None


def _migrate_pyproject_toml_to_flower_config(
    app: Path, toml_federation: str | None
) -> tuple[list[str], str | None]:
    """Migrate old TOML configuration to Flower config."""
    # Load and validate the old TOML configuration
    toml_path = app / "pyproject.toml"
    config, _, _ = load_and_validate(toml_path, check_module=False)
    if config is None:
        raise ValueError(f"Failed to load TOML configuration: {toml_path}")
    validate_federation_in_project_config(toml_federation, config)

    # Construct SuperLinkConnection
    toml_federations: dict[str, Any] = config["tool"]["flwr"]["federations"]
    migrated_conn_names: list[str] = []
    for name, toml_fed_config in toml_federations.items():
        if isinstance(toml_fed_config, dict):
            conn = parse_superlink_connection(toml_fed_config, name)
            write_superlink_connection(conn)
            migrated_conn_names.append(name)

    # Set default federation if applicable
    default_toml_federation: str | None = toml_federations.get("default")
    if default_toml_federation in toml_federations:
        set_default_superlink_connection(default_toml_federation)

    return migrated_conn_names, default_toml_federation


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
                # Preserve empty lines and comment out others
                f.write(f"# {line}" if line.strip() != "" else line)
            else:
                f.write(line)


def migrate(
    app: Path,
    toml_federation: str | None,
) -> None:
    """Migrate legacy TOML configuration to Flower config."""
    # Initialize Flower config
    init_flwr_config()

    # Print migration notice
    typer.echo(CLI_NOTICE)

    # Check if migration is applicable
    app = app.resolve()
    try:
        _check_is_migratable(app)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"Cannot migrate configuration:\n{e}") from e

    try:
        migrated_conns, default_conn = _migrate_pyproject_toml_to_flower_config(
            app, toml_federation
        )
    except Exception as e:
        raise click.ClickException(
            f"Failed to migrate legacy TOML configuration to Flower config:\n{e!r}"
        ) from e

    typer.secho("✅ Migration completed successfully!\n", fg=typer.colors.GREEN)

    # Print migrated connections
    typer.secho("Migrated SuperLink connections:", fg=typer.colors.BLUE)
    for conn_name in migrated_conns:
        typer.secho(f"  {conn_name}", fg=typer.colors.GREEN, nl=False)
        if conn_name == default_conn:
            typer.secho(" (default)", fg=typer.colors.WHITE, nl=False)
        typer.echo()

    # print usage
    typer.secho("\nYou should now use the Flower CLI as follows:")
    ctx = click.get_current_context()
    typer.secho(ctx.get_usage() + "\n", bold=True)

    _comment_out_legacy_toml_config(app)


def migrate_if_legacy_usage(
    superlink: str,
    args: list[str],
) -> None:
    """Migrate legacy TOML configuration to Flower config if legacy usage is
    detected."""
    # Trigger the same typer error when detecting unexpected extra args
    if len(args) > 1:
        raise click.UsageError(f"Got unexpected extra arguments ({' '.join(args[1:])})")

    # Skip migration if no legacy usage is detected
    if not _is_legacy_usage(superlink, args):
        return

    migrate(
        app=Path(superlink),
        toml_federation=args[0] if len(args) == 1 else None,
    )

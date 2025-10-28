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
"""Flower command line interface `build` command."""


import hashlib
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import pathspec
import tomli_w
import typer

from flwr.common.constant import (
    FAB_DATE,
    FAB_EXCLUDE_PATTERNS,
    FAB_HASH_TRUNCATION,
    FAB_INCLUDE_PATTERNS,
    FAB_MAX_SIZE,
)

from .config_utils import load as load_toml
from .config_utils import load_and_validate
from .utils import is_valid_project_name


def write_to_zip(
    zipfile_obj: zipfile.ZipFile, filename: str, contents: Union[bytes, str]
) -> zipfile.ZipFile:
    """Set a fixed date and write contents to a zip file."""
    zip_info = zipfile.ZipInfo(filename)
    zip_info.date_time = FAB_DATE
    zipfile_obj.writestr(zip_info, contents)
    return zipfile_obj


def get_fab_filename(config: dict[str, Any], fab_hash: str) -> str:
    """Get the FAB filename based on the given config and FAB hash."""
    publisher = config["tool"]["flwr"]["app"]["publisher"]
    name = config["project"]["name"]
    version = config["project"]["version"].replace(".", "-")
    fab_hash_truncated = fab_hash[:FAB_HASH_TRUNCATION]
    return f"{publisher}.{name}.{version}.{fab_hash_truncated}.fab"


# pylint: disable=too-many-locals, too-many-statements
def build(
    app: Annotated[
        Optional[Path],
        typer.Option(help="Path of the Flower App to bundle into a FAB"),
    ] = None,
) -> None:
    """Build a Flower App into a Flower App Bundle (FAB).

    You can run ``flwr build`` without any arguments to bundle the app located in the
    current directory. Alternatively, you can you can specify a path using the ``--app``
    option to bundle an app located at the provided path. For example:

    ``flwr build --app ./apps/flower-hello-world``.
    """
    if app is None:
        app = Path.cwd()

    app = app.resolve()
    if not app.is_dir():
        typer.secho(
            f"❌ The path {app} is not a valid path to a Flower app.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if not is_valid_project_name(app.name):
        typer.secho(
            f"❌ The project name {app.name} is invalid, "
            "a valid project name must start with a letter, "
            "and can only contain letters, digits, and hyphens.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    config, errors, warnings = load_and_validate(app / "pyproject.toml")
    if config is None:
        typer.secho(
            "Project configuration could not be loaded.\npyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    # Build FAB
    fab_bytes, fab_hash, _ = build_fab(app)

    # Get the name of the zip file
    fab_filename = get_fab_filename(config, fab_hash)

    # Write the FAB
    Path(fab_filename).write_bytes(fab_bytes)

    typer.secho(
        f"🎊 Successfully built {fab_filename}", fg=typer.colors.GREEN, bold=True
    )


def build_fab(app: Path) -> tuple[bytes, str, dict[str, Any]]:
    """Build a FAB in memory and return the bytes, hash, and config.

    This function assumes that the provided path points to a valid Flower app and
    bundles it into a FAB without performing additional validation.

    Parameters
    ----------
    app : Path
        Path to the Flower app to bundle into a FAB.

    Returns
    -------
    tuple[bytes, str, dict[str, Any]]
        A tuple containing:
        - the FAB as bytes
        - the SHA256 hash of the FAB
        - the project configuration (with the 'federations' field removed)
    """
    app = app.resolve()

    # Load the pyproject.toml file
    config = load_toml(app / "pyproject.toml")
    if config is None:
        raise ValueError("Project configuration could not be loaded.")

    # Remove the 'federations' field if it exists
    if (
        "tool" in config
        and "flwr" in config["tool"]
        and "federations" in config["tool"]["flwr"]
    ):
        del config["tool"]["flwr"]["federations"]

    # Load include spec
    gitignore_content = None
    if (app / ".gitignore").is_file():
        # Load .gitignore rules if present
        gitignore_content = (app / ".gitignore").read_bytes()
    exclude_spec = get_fab_exclude_pathspec(gitignore_content)

    # Load include spec
    include_spec = get_fab_include_pathspec()

    # Search for all files in the app directory
    all_files = [
        f
        for f in app.rglob("*")
        if include_spec.match_file(f) and not exclude_spec.match_file(f)
    ]
    all_files.sort()

    # Create a zip file in memory
    list_file_content = ""

    fab_buffer = BytesIO()
    with zipfile.ZipFile(fab_buffer, "w", zipfile.ZIP_DEFLATED) as fab_file:
        # Add pyproject.toml
        write_to_zip(fab_file, "pyproject.toml", tomli_w.dumps(config))

        for file_path in all_files:
            # Read the file content manually
            file_contents = file_path.read_bytes()

            archive_path = str(file_path.relative_to(app)).replace("\\", "/")
            write_to_zip(fab_file, archive_path, file_contents)

            # Calculate file info
            sha256_hash = hashlib.sha256(file_contents).hexdigest()
            file_size_bits = len(file_contents) * 8  # size in bits
            list_file_content += f"{archive_path},{sha256_hash},{file_size_bits}\n"

        # Add CONTENT and CONTENT.jwt to the zip file
        write_to_zip(fab_file, ".info/CONTENT", list_file_content)

    fab_bytes = fab_buffer.getvalue()
    if len(fab_bytes) > FAB_MAX_SIZE:
        raise ValueError(
            f"FAB size exceeds maximum allowed size of {FAB_MAX_SIZE:,} bytes."
            "To reduce the package size, consider ignoring unnecessary files "
            "via your `.gitignore` file or excluding them from the build."
        )

    fab_hash = hashlib.sha256(fab_bytes).hexdigest()

    return fab_bytes, fab_hash, config


def build_pathspec(patterns: Iterable[str]) -> pathspec.PathSpec:
    """Build a PathSpec from a list of patterns."""
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def get_fab_include_pathspec() -> pathspec.PathSpec:
    """Get the PathSpec for files to include in a FAB."""
    return build_pathspec(FAB_INCLUDE_PATTERNS)


def get_fab_exclude_pathspec(gitignore_content: Optional[bytes]) -> pathspec.PathSpec:
    """Get the PathSpec for files to exclude from a FAB.

    If gitignore_content is provided, its patterns will be combined with the default
    exclude patterns.
    """
    patterns = list(FAB_EXCLUDE_PATTERNS)
    if gitignore_content:
        patterns += gitignore_content.decode("UTF-8").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

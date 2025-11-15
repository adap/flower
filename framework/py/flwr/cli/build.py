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
from typing import Annotated, Any

import pathspec
import tomli
import tomli_w
import typer

from flwr.common.constant import (
    FAB_CONFIG_FILE,
    FAB_DATE,
    FAB_EXCLUDE_PATTERNS,
    FAB_HASH_TRUNCATION,
    FAB_INCLUDE_PATTERNS,
    FAB_MAX_SIZE,
)

from .config_utils import load_and_validate
from .utils import is_valid_project_name


def write_to_zip(
    zipfile_obj: zipfile.ZipFile, filename: str, contents: bytes | str
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
        Path | None,
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
    fab_bytes = build_fab_from_disk(app)

    # Calculate hash for filename
    fab_hash = hashlib.sha256(fab_bytes).hexdigest()

    # Get the name of the zip file
    fab_filename = get_fab_filename(config, fab_hash)

    # Write the FAB
    Path(fab_filename).write_bytes(fab_bytes)

    typer.secho(
        f"🎊 Successfully built {fab_filename}", fg=typer.colors.GREEN, bold=True
    )


def build_fab_from_disk(app: Path) -> bytes:
    """Build a FAB from files on disk and return the FAB as bytes.

    This function reads files from disk and bundles them into a FAB.

    Parameters
    ----------
    app : Path
        Path to the Flower app to bundle into a FAB.

    Returns
    -------
    bytes
        The FAB as bytes.
    """
    app = app.resolve()

    # Collect all files recursively (including pyproject.toml and .gitignore)
    all_files = [f for f in app.rglob("*") if f.is_file()]

    # Create dict mapping relative paths to Path objects
    files_dict: dict[str, bytes | Path] = {
        # Ensure consistent path separators across platforms
        str(file_path.relative_to(app)).replace("\\", "/"): file_path
        for file_path in all_files
    }

    # Build FAB from the files dict
    return build_fab_from_files(files_dict)


def build_fab_from_files(files: dict[str, bytes | Path]) -> bytes:
    r"""Build a FAB from in-memory files and return the FAB as bytes.

    This is the core FAB building function that works with in-memory data.
    It accepts either bytes or Path objects as file contents, applies filtering
    rules (include/exclude patterns), and builds the FAB.

    Parameters
    ----------
    files : dict[str, Union[bytes, Path]]
        Dictionary mapping relative file paths to their contents.
        - Keys: Relative paths (strings)
        - Values: Either bytes (file contents) or Path (will be read)
        Must include "pyproject.toml" and optionally ".gitignore".

    Returns
    -------
    bytes
        The FAB as bytes.

    Examples
    --------
    Build a FAB from in-memory files::

        files = {
            "pyproject.toml": b"[project]\nname = 'myapp'\n...",
            ".gitignore": b"*.pyc\n__pycache__/\n",
            "src/client.py": Path("/path/to/client.py"),
            "src/server.py": b"print('hello')",
            "README.md": b"# My App\n",
        }
        fab_bytes = build_fab_from_files(files)
    """

    def to_bytes(content: bytes | Path) -> bytes:
        return content.read_bytes() if isinstance(content, Path) else content

    # Extract, load, and parse pyproject.toml
    if FAB_CONFIG_FILE not in files:
        raise ValueError(f"{FAB_CONFIG_FILE} not found in files")
    pyproject_content = to_bytes(files[FAB_CONFIG_FILE])
    config = tomli.loads(pyproject_content.decode("utf-8"))

    # Remove the 'federations' field if it exists
    if (
        "tool" in config
        and "flwr" in config["tool"]
        and "federations" in config["tool"]["flwr"]
    ):
        del config["tool"]["flwr"]["federations"]

    # Extract and load .gitignore if present
    gitignore_content = None
    if ".gitignore" in files:
        gitignore_content = to_bytes(files[".gitignore"])

    # Get exclude and include specs
    exclude_spec = get_fab_exclude_pathspec(gitignore_content)
    include_spec = get_fab_include_pathspec()

    # Filter files based on include/exclude specs
    filtered_paths = [
        path.replace("\\", "/")  # Ensure consistent path separators across platforms
        for path in files.keys()
        if include_spec.match_file(path) and not exclude_spec.match_file(path)
    ]
    filtered_paths.sort()  # Sort for deterministic output

    # Create a zip file in memory
    list_file_content = ""

    fab_buffer = BytesIO()
    with zipfile.ZipFile(fab_buffer, "w", zipfile.ZIP_DEFLATED) as fab_file:
        # Add pyproject.toml
        write_to_zip(fab_file, FAB_CONFIG_FILE, tomli_w.dumps(config))

        for file_path in filtered_paths:

            # Get file contents as bytes
            file_content = to_bytes(files[file_path])

            # Write file to FAB
            write_to_zip(fab_file, file_path, file_content)

            # Calculate file info for CONTENT manifest
            sha256_hash = hashlib.sha256(file_content).hexdigest()
            file_size_bits = len(file_content) * 8  # size in bits
            list_file_content += f"{file_path},{sha256_hash},{file_size_bits}\n"

        # Add CONTENT manifest to the zip file
        write_to_zip(fab_file, ".info/CONTENT", list_file_content)

    fab_bytes = fab_buffer.getvalue()

    # Validate FAB size
    if len(fab_bytes) > FAB_MAX_SIZE:
        raise ValueError(
            f"FAB size exceeds maximum allowed size of {FAB_MAX_SIZE:,} bytes. "
            "To reduce the package size, consider ignoring unnecessary files "
            "via your `.gitignore` file or excluding them from the build."
        )

    return fab_bytes


def build_pathspec(patterns: Iterable[str]) -> pathspec.PathSpec:
    """Build a PathSpec from a list of patterns."""
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def get_fab_include_pathspec() -> pathspec.PathSpec:
    """Get the PathSpec for files to include in a FAB."""
    return build_pathspec(FAB_INCLUDE_PATTERNS)


def get_fab_exclude_pathspec(gitignore_content: bytes | None) -> pathspec.PathSpec:
    """Get the PathSpec for files to exclude from a FAB.

    If gitignore_content is provided, its patterns will be combined with the default
    exclude patterns.
    """
    patterns = list(FAB_EXCLUDE_PATTERNS)
    if gitignore_content:
        patterns += gitignore_content.decode("UTF-8").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

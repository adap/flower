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
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import click
import pathspec
import tomli
import tomli_w
import typer

from flwr.common.config import check_pattern_list_value
from flwr.common.constant import (
    FAB_CONFIG_FILE,
    FAB_DATE,
    FAB_EXCLUDE_KEY,
    FAB_EXCLUDE_PATTERNS,
    FAB_HASH_TRUNCATION,
    FAB_INCLUDE_KEY,
    FAB_INCLUDE_PATTERNS,
    FAB_MAX_SIZE,
)
from flwr.supercore.fab_format_version import (
    FabFormatMetadata,
    normalize_and_validate_fab_format,
    validate_fab_files_for_format,
)

from .config_utils import load_and_validate
from .utils import build_pathspec, is_valid_project_name


def write_to_zip(
    zipfile_obj: zipfile.ZipFile, filename: str, contents: bytes | str
) -> zipfile.ZipFile:
    """Set a fixed date and write contents to a zip file.

    Parameters
    ----------
    zipfile_obj : zipfile.ZipFile
        The ZipFile object to write to.
    filename : str
        Name of the file within the zip archive.
    contents : bytes | str
        The file contents to write.

    Returns
    -------
    ZipFile
        The modified ZipFile object.
    """
    zip_info = zipfile.ZipInfo(filename)
    zip_info.date_time = FAB_DATE
    zipfile_obj.writestr(zip_info, contents)
    return zipfile_obj


def get_fab_filename(config: dict[str, Any], fab_hash: str) -> str:
    """Get the FAB filename based on the given config and FAB hash.

    Parameters
    ----------
    config : dict[str, Any]
        The Flower App configuration dictionary.
    fab_hash : str
        The SHA-256 hash of the FAB file.

    Returns
    -------
    str
        The formatted FAB filename in the pattern:
        <publisher>.<name>.<version>.<hash_prefix>.fab
    """
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

    app = app.expanduser().resolve()
    if not app.is_dir():
        raise click.ClickException(
            f"The path {app} is not a valid path to a Flower app."
        )

    if not is_valid_project_name(app.name):
        raise click.ClickException(
            f"The Flower App name {app.name} is invalid, "
            "a valid app name must start with a letter, "
            "and can only contain letters, digits, and hyphens."
        )

    try:
        config, warnings = load_and_validate(app / "pyproject.toml")
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if warnings:
        typer.secho(
            "Flower App configuration (pyproject.toml) is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.YELLOW,
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
    fab_bytes, _ = build_fab_from_files(files_dict)
    return fab_bytes


def build_fab_from_files(
    files: dict[str, bytes | Path],
) -> tuple[bytes, FabFormatMetadata]:
    r"""Build a FAB from in-memory files and return the FAB plus metadata.

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
    tuple[bytes, FabFormatMetadata]
        The FAB as bytes together with normalized compatibility metadata.
        The metadata is consumed by platform-api during publish to persist
        compatibility fields derived from this shared build validation logic.

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
        fab_bytes, metadata = build_fab_from_files(files)
    """

    def _to_bytes(content: bytes | Path) -> bytes:
        return content.read_bytes() if isinstance(content, Path) else content

    def _add_to_fab(
        fab_file: zipfile.ZipFile,
        path: str,
        content: bytes,
    ) -> str:
        """Write a file to the FAB and return its CONTENT manifest line.

        Parameters
        ----------
        fab_file : zipfile.ZipFile
            The ZipFile object to write to.
        path : str
            The file path within the FAB.
        content : bytes
            The file contents as bytes.

        Returns
        -------
        str
            A CONTENT manifest line: "path,sha256,size_bits"
        """
        write_to_zip(fab_file, path, content)
        sha256_hash = hashlib.sha256(content).hexdigest()
        file_size_bits = len(content) * 8
        return f"{path},{sha256_hash},{file_size_bits}"

    # Extract, load, and parse pyproject.toml
    if FAB_CONFIG_FILE not in files:
        raise ValueError(f"{FAB_CONFIG_FILE} not found in files")
    pyproject_content = _to_bytes(files[FAB_CONFIG_FILE])
    config = tomli.loads(pyproject_content.decode("utf-8"))
    metadata = normalize_and_validate_fab_format(config)

    # Remove the 'federations' field if it exists
    if (
        "tool" in config
        and "flwr" in config["tool"]
        and "federations" in config["tool"]["flwr"]
    ):
        del config["tool"]["flwr"]["federations"]

    # Filter files based on user patterns and built-in constraints.
    filtered_paths = get_filtered_fab_paths(files, config)
    filtered_paths.sort()  # Sort for deterministic output
    validate_fab_files_for_format(config, filtered_paths)

    # Build FAB with CONTENT manifest
    fab_buffer = BytesIO()
    with zipfile.ZipFile(fab_buffer, "w", zipfile.ZIP_DEFLATED) as fab_file:
        # Add pyproject.toml and collect manifest entries
        pyproject_bytes = tomli_w.dumps(config).encode("utf-8")
        manifest_lines = [_add_to_fab(fab_file, FAB_CONFIG_FILE, pyproject_bytes)]

        # Add remaining files and collect their manifest entries
        for file_path in filtered_paths:
            file_content = _to_bytes(files[file_path])
            manifest_lines.append(_add_to_fab(fab_file, file_path, file_content))

        # Write CONTENT manifest to the zip file
        write_to_zip(fab_file, ".info/CONTENT", "\n".join(manifest_lines))

    fab_bytes = fab_buffer.getvalue()

    # Validate FAB size
    if len(fab_bytes) > FAB_MAX_SIZE:
        raise ValueError(
            f"FAB size exceeds maximum allowed size of {FAB_MAX_SIZE:,} bytes. "
            f"To reduce package size, narrow `{FAB_INCLUDE_KEY}` or add "
            f"`{FAB_EXCLUDE_KEY}` patterns in [tool.flwr.app]."
        )

    # Returned metadata is consumed by platform during publish.
    return fab_bytes, metadata


def get_user_fab_patterns(
    config: dict[str, Any],
) -> tuple[list[str] | None, list[str] | None]:
    """Return user-defined FAB include/exclude patterns.

    Returns ``None`` for a key that is absent from the config, or the
    non-empty pattern list itself. Raises ``ValueError`` if a key is
    present but set to an empty list.
    """
    app_conf = config.get("tool", {}).get("flwr", {}).get("app", {})
    if not isinstance(app_conf, dict):
        return None, None

    def _get_pattern_list(key: str) -> list[str] | None:
        if key not in app_conf:
            return None
        value: list[str] = app_conf[key]
        error = check_pattern_list_value(value, key)
        if error:
            raise ValueError(error)
        return value

    return (_get_pattern_list(FAB_INCLUDE_KEY), _get_pattern_list(FAB_EXCLUDE_KEY))


def get_filtered_fab_paths(
    files: dict[str, bytes | Path],
    config: dict[str, Any],
) -> list[str]:
    """Compute final FAB file list using user patterns and non-overridable defaults."""
    normalized_paths = [path.replace("\\", "/") for path in files.keys()]
    built_in_include_spec = build_pathspec(FAB_INCLUDE_PATTERNS)
    built_in_exclude_spec = build_pathspec(FAB_EXCLUDE_PATTERNS)

    user_include_patterns, user_exclude_patterns = get_user_fab_patterns(config)
    user_include_spec = (
        build_pathspec(user_include_patterns) if user_include_patterns else None
    )
    user_exclude_spec = (
        build_pathspec(user_exclude_patterns) if user_exclude_patterns else None
    )
    messages: list[tuple[str, str]] = []
    messages.extend(
        _collect_unresolved_pattern_messages(
            user_include_patterns or [], normalized_paths, FAB_INCLUDE_KEY
        )
    )
    messages.extend(
        _collect_unresolved_pattern_messages(
            user_exclude_patterns or [], normalized_paths, FAB_EXCLUDE_KEY
        )
    )

    # Candidate set: user include matches, or all files if no include patterns provided.
    candidate_paths = (
        [path for path in normalized_paths if user_include_spec.match_file(path)]
        if user_include_spec
        else normalized_paths
    )

    built_in_constrained_paths = [
        path
        for path in candidate_paths
        if built_in_include_spec.match_file(path)
        and not built_in_exclude_spec.match_file(path)
    ]
    final_paths = (
        [
            path
            for path in built_in_constrained_paths
            if not user_exclude_spec.match_file(path)
        ]
        if user_exclude_spec
        else list(built_in_constrained_paths)
    )
    messages.extend(
        _collect_pattern_conflict_messages(
            user_include_spec=user_include_spec,
            user_exclude_spec=user_exclude_spec,
            candidate_paths=candidate_paths,
            built_in_constrained_paths=built_in_constrained_paths,
        )
    )
    _emit_filter_messages(messages)
    return final_paths


def _collect_unresolved_pattern_messages(
    patterns: list[str], file_paths: list[str], key_name: str
) -> list[tuple[str, str]]:
    """Collect warning messages for unresolved user-defined patterns."""
    messages: list[tuple[str, str]] = []
    for pattern in patterns:
        try:
            pattern_spec = build_pathspec([pattern])
        except Exception as err:  # pylint: disable=broad-except
            messages.append(
                (
                    "Warning",
                    f'ignoring unresolved pattern in "{key_name}": '
                    f'"{pattern}" ({err})',
                )
            )
            continue

        if not any(pattern_spec.match_file(path) for path in file_paths):
            messages.append(
                (
                    "Warning",
                    f'pattern in "{key_name}" did not match any files: "{pattern}"',
                )
            )
    return messages


def _collect_pattern_conflict_messages(
    user_include_spec: pathspec.PathSpec | None,
    user_exclude_spec: pathspec.PathSpec | None,
    candidate_paths: list[str],
    built_in_constrained_paths: list[str],
) -> list[tuple[str, str]]:
    """Collect warning messages for include/exclude and built-in conflicts."""
    messages: list[tuple[str, str]] = []
    if user_include_spec and user_exclude_spec:
        overlap = [
            path
            for path in candidate_paths
            if user_include_spec.match_file(path) and user_exclude_spec.match_file(path)
        ]
        if overlap:
            messages.append(
                (
                    "Warning",
                    f'"{FAB_INCLUDE_KEY}" and "{FAB_EXCLUDE_KEY}" overlap for '
                    f"{len(overlap)} file(s); exclusion takes precedence.",
                )
            )

    built_in_removed = len(candidate_paths) - len(built_in_constrained_paths)
    if user_include_spec and built_in_removed > 0:
        messages.append(
            (
                "Warning",
                f'{built_in_removed} file(s) matched "{FAB_INCLUDE_KEY}" but '
                "were removed by non-overridable built-in FAB constraints.",
            )
        )
    return messages


def _emit_filter_messages(messages: list[tuple[str, str]]) -> None:
    """Emit filter notes/warnings with consistent CLI formatting."""
    for level, message in messages:
        typer.secho(
            f"{level}: {message}",
            fg=typer.colors.YELLOW,
            bold=True,
        )

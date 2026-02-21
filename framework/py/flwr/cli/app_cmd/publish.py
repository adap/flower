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
"""Flower command line interface `app publish` command."""


from contextlib import ExitStack
from pathlib import Path
from typing import IO, Annotated

import click
import requests
import typer
from requests import Response

from flwr.supercore.constant import (
    APP_PUBLISH_EXCLUDE_PATTERNS,
    APP_PUBLISH_INCLUDE_PATTERNS,
    MAX_DIR_DEPTH,
    MAX_FILE_BYTES,
    MAX_FILE_COUNT,
    MAX_TOTAL_BYTES,
    MIME_MAP,
    PLATFORM_API_URL,
    SUPERGRID_ADDRESS,
    UTF8,
)
from flwr.supercore.version import package_version as flwr_version

from ..auth_plugin.oidc_cli_plugin import OidcCliPlugin
from ..utils import (
    build_pathspec,
    load_cli_auth_plugin_from_connection,
    load_gitignore_patterns,
)


# pylint: disable=too-many-locals
def publish(
    app: Annotated[
        Path,
        typer.Argument(
            help="Project directory to upload (defaults to current directory)."
        ),
    ] = Path("."),
) -> None:
    """Publish a Flower App to the Flower Platform.

    This command uploads your app project to the Flower Platform. Files are filtered
    based on .gitignore patterns and allowed file extensions.
    """
    auth_plugin = load_cli_auth_plugin_from_connection(SUPERGRID_ADDRESS)
    auth_plugin.load_tokens()
    if not isinstance(auth_plugin, OidcCliPlugin) or not auth_plugin.access_token:
        raise click.ClickException("Please log in before publishing app.")

    # Load token from the plugin
    token = auth_plugin.access_token

    # Collect & validate app files
    file_paths = _collect_file_paths(app)
    _validate_files(file_paths)

    # Build and POST multipart
    with ExitStack() as stack:
        files_param = _build_multipart_files_param(app, file_paths, stack)
        try:
            resp = _post_files(files_param, token)
        except requests.RequestException as err:
            raise click.ClickException(f"Network error: {err}") from err

    if resp.ok:
        typer.secho("🎊 Upload successful", fg=typer.colors.GREEN, bold=True)
        return  # success

    # Error path:
    msg = f"Upload failed with status {resp.status_code}"
    if resp.text:
        msg += f": {resp.text}"
    raise click.ClickException(msg)


def _depth_of(relative_path_to_root: Path) -> int:
    """Return the directory depth of `relative_path_to_root`.

    Depth is the number of directory components, excluding the filename. For example,
    `"a/b/c.py"` has depth 2.
    """
    return max(0, len(relative_path_to_root.parts) - 1)


def _detect_mime(path: Path) -> str:
    """Detect files' MIME."""
    return MIME_MAP.get(path.suffix.lower(), "text/plain; charset=utf-8")


def _collect_file_paths(root: Path) -> list[Path]:
    """Return list of file paths that match include/exclude patterns."""
    # Build include/exclude pathspecs
    # Note: This should be a temporary solution until we have a complete mechanism
    # for configurable inclusion and exclusion rules.
    # Note: Unlike Git, we do not support nested .gitignore files in subdirectories.
    gitignore_patterns = tuple(load_gitignore_patterns(root / ".gitignore"))
    exclude_pathspec = build_pathspec(gitignore_patterns + APP_PUBLISH_EXCLUDE_PATTERNS)
    include_pathspec = build_pathspec(APP_PUBLISH_INCLUDE_PATTERNS)

    # Walk the directory tree
    file_paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Skip excluded or not included files
        # Note: pathspec requires POSIX style relative paths
        relative_path = path.relative_to(root)
        posix = relative_path.as_posix()
        if exclude_pathspec.match_file(posix) or not include_pathspec.match_file(posix):
            typer.echo(typer.style(f"Skip: {path}", fg=typer.colors.YELLOW))
            continue

        # Check max depth
        if _depth_of(relative_path) > MAX_DIR_DEPTH:
            raise click.ClickException(
                f"'{path}' exceeds the maximum directory depth of {MAX_DIR_DEPTH}."
            )

        file_paths.append(path)

    # Sort for deterministic ordering
    file_paths.sort(key=lambda path: path.as_posix())
    return file_paths


def _validate_files(file_paths: list[Path]) -> None:
    """Validate files against upload constraints.

    Checks file count, individual file size, total size, and UTF-8 encoding.
    """
    if len(file_paths) == 0:
        raise click.ClickException(
            "Nothing to upload: no files matched after applying .gitignore and "
            "allowed extensions."
        )

    if len(file_paths) > MAX_FILE_COUNT:
        raise click.ClickException(
            f"Too many files: {len(file_paths)} > allowed maximum of {MAX_FILE_COUNT}."
        )

    # Calculate files size
    total_size = 0
    for path in file_paths:
        file_size = path.stat().st_size
        total_size += file_size

        # Check single file size
        if file_size > MAX_FILE_BYTES:
            raise click.ClickException(
                f"File too large: '{path.as_posix()}' is {file_size:,} bytes, "
                f"exceeding the per-file limit of {MAX_FILE_BYTES:,} bytes."
            )

        # Ensure we can decode as UTF-8.
        try:
            path.read_text(encoding=UTF8)
        except UnicodeDecodeError as err:
            raise click.ClickException(
                f"Encoding error: '{path}' is not UTF-8 encoded."
            ) from err

    # Check total files size
    if total_size > MAX_TOTAL_BYTES:
        raise click.ClickException(
            "Total size of all files is too large: "
            f"{total_size:,} bytes > {MAX_TOTAL_BYTES:,} bytes."
        )

    # Print validation passed prompt
    typer.echo(typer.style("✅ Validation passed", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"{len(file_paths)} files, {total_size:,} bytes in total")


def _build_multipart_files_param(
    root: Path,
    file_paths: list[Path],
    stack: ExitStack,
) -> list[tuple[str, tuple[str, IO[bytes], str]]]:
    """Build multipart/form-data files parameter for HTTP upload.

    Returns list of tuples: (field_name, (filename, file_object, content_type)).
    File handles are registered with ExitStack for proper cleanup.
    """
    form: list[tuple[str, tuple[str, IO[bytes], str]]] = []
    for path in file_paths:
        # Detect MIME (content type)
        mime = _detect_mime(path)

        # Open file and register with ExitStack
        # pylint: disable-next=consider-using-with
        fobj = stack.enter_context(open(path.resolve(), "rb"))
        typer.echo(f"Attach {path} ({mime}, {path.stat().st_size:,} bytes)")

        # Get relative POSIX path
        relative_posix = path.relative_to(root).as_posix()

        # Append to form data (key, (filename, fileobj, mime))
        form.append(("files", (relative_posix, fobj, mime)))
    return form


def _post_files(
    files_param: list[tuple[str, tuple[str, IO[bytes], str]]],
    token: str,
) -> Response:
    """POST multipart with one part per file."""
    url = f"{PLATFORM_API_URL}/hub/apps/publish"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"flwr_version": flwr_version}

    resp = requests.post(
        url, files=files_param, headers=headers, json=body, timeout=120
    )
    return resp

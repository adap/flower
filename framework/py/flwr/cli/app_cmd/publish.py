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


from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Annotated

import pathspec
import requests
import typer
from requests import Response

from flwr.common.constant import CREDENTIALS_DIR, FLWR_DIR
from flwr.supercore.constant import (
    ALLOWED_EXTS,
    MAX_DIR_DEPTH,
    MAX_FILE_BYTES,
    MAX_FILE_COUNT,
    MAX_TOTAL_BYTES,
    MIME_MAP,
    PLATFORM_API_URL,
    UTF8,
)

from ..utils import (
    build_pathspec,
    get_exclude_pathspec,
    to_bytes,
    validate_credentials_content,
)


def publish(
    app: Annotated[
        Path,
        typer.Argument(
            help="Project directory to upload (defaults to current directory)."
        ),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(
            help="Name of the federation used for login before publishing app."
        ),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="Bearer token for Platform API.",
        ),
    ] = None,
) -> None:
    """Upload all project files to the Platform API using multipart/form-data."""
    # Check the credentials path
    if not token:
        if not federation:
            typer.secho(
                "❌ Please specify the federation used for "
                "login before publishing app.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        creds_path = app.absolute() / FLWR_DIR / CREDENTIALS_DIR / f"{federation}.json"
        if not creds_path.is_file():
            typer.secho(
                "❌ Please log in before publishing app.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Load and validate credentials
        token = validate_credentials_content(creds_path)

    # Collect & validate app files
    files = _collect_files(app)
    _validate_files(files)

    # Build and POST multipart
    with ExitStack() as stack:
        files_param = _build_multipart_files_param(files, stack)
        try:
            resp = _post_files(files_param, token)
        except requests.RequestException as err:
            typer.secho(f"❌ Network error: {err}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from err

    if resp.ok:
        typer.secho("🎊 Upload successful", fg=typer.colors.GREEN, bold=True)
        return  # success

    # Error path:
    msg = f"❌ Upload failed with status {resp.status_code}"
    if resp.text:
        msg += f": {resp.text}"
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _load_gitignore(root: Path) -> bytes | None:
    """Load gitignore file."""
    gitignore_path = root / ".gitignore"
    if gitignore_path.is_file():
        return to_bytes(gitignore_path)
    return None


def _compile_gitignore(root: Path) -> Callable[[Path], bool]:
    """Return a callable `ignored(path: Path) -> bool`.

    Paths are evaluated relative to `root` with POSIX separators.
    Always ignores the Flower internal directory (`FLWR_DIR`).
    """
    gitignore_content = _load_gitignore(root)
    # Always ignore the internal Flower directory
    flwr_dir_relative = (
        (root.absolute() / FLWR_DIR).relative_to(root.absolute()).as_posix()
    )
    if not flwr_dir_relative.endswith("/"):
        flwr_dir_relative += "/"

    # Create PathSpec
    exclude_spec = get_exclude_pathspec([flwr_dir_relative], gitignore_content)

    def ignored(path: Path) -> bool:
        relative_path = path.relative_to(root).as_posix()
        return exclude_spec.match_file(relative_path)

    return ignored


def _get_publish_exclude_pathspec(root: Path) -> pathspec.PathSpec:
    """Get the PathSpec for files to exclude based on .gitignore."""
    # Exclude .flwr/ directory by default
    exclude_patterns = [f"{FLWR_DIR}/"]
    return get_exclude_pathspec()


def _get_include_extensions_pathspec() -> pathspec.PathSpec:
    """Get the PathSpec for files to include."""
    return build_pathspec(ALLOWED_EXTS)


def _depth_of(relative_path: Path) -> int:
    """Return depth that is number of parts (directories) in the relative path
    (excluding filename).

    Example: "a/b/c.py" -> depth 3
    Interpret "directory depth" as number of directories: len(parts) - 1
    """
    return max(0, len(relative_path.parts) - 1)


def _detect_mime(path: Path) -> str:
    """Detect files' MIME."""
    return MIME_MAP.get(path.suffix.lower(), "text/plain; charset=utf-8")


def _collect_files(root: Path) -> list[tuple[Path, Path]]:
    """Return list of (absolute_path, relative_path) that pass allowed list + ignore
    logic."""
    ignored = _compile_gitignore(root)
    files: list[tuple[Path, Path]] = []

    for absolute_path in root.rglob("*"):
        try:
            if ignored(absolute_path):
                if absolute_path.is_file():
                    typer.echo(
                        typer.style(
                            f"Skip (gitignore): {absolute_path}", fg=typer.colors.YELLOW
                        )
                    )
                continue
        except Exception:  # pylint: disable=broad-exception-caught
            # If ignore callable crashes for some weird FS entry, just continue safely
            continue

        if absolute_path.is_dir():
            continue

        # Only include allowed extensions
        include_extensions_spec = _get_include_extensions_pathspec()
        relative_path = absolute_path.relative_to(root)
        if not include_extensions_spec.match_file(relative_path.as_posix()):
            typer.echo(
                typer.style(f"Skip (ext): {absolute_path}", fg=typer.colors.YELLOW)
            )
            continue

        # Check max depth
        if _depth_of(relative_path) > MAX_DIR_DEPTH:
            typer.secho(
                f"Error: '{relative_path.as_posix()}' "
                f"exceeds the maximum directory depth "
                f"of {MAX_DIR_DEPTH}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        files.append((absolute_path, relative_path))

    # Sort for deterministic ordering
    files.sort(key=lambda pr: pr[1].as_posix())
    return files


def _validate_files(files: list[tuple[Path, Path]]) -> None:
    """Validate files based on spec."""
    if len(files) == 0:
        typer.secho(
            "Nothing to upload: no files matched after applying .gitignore and "
            "allowed extensions.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if len(files) > MAX_FILE_COUNT:
        typer.secho(
            f"Too many files: {len(files)} > allowed maximum of {MAX_FILE_COUNT}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Calculate files size
    total = 0
    for absolute_path, relative_path in files:
        size = absolute_path.stat().st_size
        total += size

        # Check single file size
        if size > MAX_FILE_BYTES:
            typer.secho(
                f"File too large: '{relative_path.as_posix()}' is {size} bytes, "
                f"exceeding the per-file limit of {MAX_FILE_BYTES} bytes.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        # Ensure we can decode as UTF-8.
        try:
            absolute_path.read_text(encoding=UTF8)
        except UnicodeDecodeError as err:
            typer.secho(
                f"Encoding error: '{relative_path.as_posix()}' is not UTF-8 encoded.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from err

    # Check total files size
    if total > MAX_TOTAL_BYTES:
        typer.secho(
            f"Total payload too large: {total} bytes > {MAX_TOTAL_BYTES} bytes.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Print validation passed prompt
    typer.echo(typer.style("✅ Validation passed", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"{len(files)} files, {total} bytes total")


def _build_multipart_files_param(
    files: list[tuple[Path, Path]],
    stack: ExitStack,
) -> list[tuple[str, tuple[str, IO[bytes], str]]]:
    """Return a list suitable for requests.post(files=...) and register file handles
    with ExitStack."""
    form: list[tuple[str, tuple[str, IO[bytes], str]]] = []
    for absolute_path, relative_path in files:
        relative_posix = relative_path.as_posix()
        mime = _detect_mime(absolute_path)

        fobj = stack.enter_context(
            open(absolute_path, "rb")  # pylint: disable=consider-using-with
        )
        typer.echo(
            f"Attach {relative_posix} ({mime}, {absolute_path.stat().st_size} B)"
        )

        form.append(("files", (relative_posix, fobj, mime)))
    return form


def _post_files(
    files_param: list[tuple[str, tuple[str, IO[bytes], str]]],
    token: str,
) -> Response:
    """POST multipart with one part per file."""
    url = f"{PLATFORM_API_URL}/hub/apps/publish"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.post(url, files=files_param, headers=headers, timeout=120)
    return resp

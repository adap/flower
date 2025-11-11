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


import json
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Annotated, Callable, Optional

import pathspec
import requests
import typer
from requests import Response

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    CREDENTIALS_DIR,
    FLWR_DIR,
    REFRESH_TOKEN_KEY,
)
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


def _load_gitignore(root: Path) -> Optional[Iterable[str]]:
    """Load gitignore file."""
    gi = root / ".gitignore"
    if gi.is_file():
        return gi.read_text(encoding="utf-8").splitlines()
    return None


def _compile_gitignore(root: Path) -> Callable[[Path], bool]:
    """Return a callable `ignored(path: Path) -> bool`.

    Paths are evaluated relative to `root` with POSIX separators.
    Always ignores the Flower internal directory (`FLWR_DIR`).
    """
    patterns = list(_load_gitignore(root) or [])

    # Always ignore the internal Flower directory
    flwr_dir_rel = (root.absolute() / FLWR_DIR).relative_to(root.absolute()).as_posix()
    if not flwr_dir_rel.endswith("/"):
        flwr_dir_rel += "/"
    patterns.append(flwr_dir_rel)

    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def ignored(p: Path) -> bool:
        rel = p.relative_to(root).as_posix()
        # pathspec expects directory entries with trailing slash for dir matches
        if p.is_dir() and not rel.endswith("/"):
            rel += "/"
        return spec.match_file(rel)

    return ignored


def _depth_of(relpath: Path) -> int:
    """Return depth that is number of parts (directories) in the relative path
    (excluding filename).

    Example: "a/b/c.py" -> depth 3
    Interpret "directory depth" as number of directories: len(parts) - 1
    """
    parts = relpath.as_posix().split("/")
    return max(0, len(parts) - 1)


def _detect_mime(path: Path) -> str:
    """Detect files' MIME."""
    return MIME_MAP.get(path.suffix.lower(), "text/plain; charset=utf-8")


def _collect_files(root: Path) -> list[tuple[Path, Path]]:
    """Return list of (abs_path, rel_path) that pass allowed list + ignore logic."""
    ignored = _compile_gitignore(root)
    files: list[tuple[Path, Path]] = []

    for p in root.rglob("*"):
        try:
            if ignored(p):
                if p.is_file():
                    typer.echo(
                        typer.style(f"Skip (gitignore): {p}", fg=typer.colors.YELLOW)
                    )
                continue
        except Exception:  # pylint: disable=broad-exception-caught
            # If ignore callable crashes for some weird FS entry, just continue safely
            continue

        if p.is_dir():
            continue

        # Only include allowed extensions
        if p.suffix.lower() not in ALLOWED_EXTS:
            typer.echo(typer.style(f"Skip (ext): {p}", fg=typer.colors.YELLOW))
            continue

        # Check max depth
        rel = p.relative_to(root)
        if _depth_of(rel) > MAX_DIR_DEPTH:
            typer.secho(
                f"Error: '{rel.as_posix()}' exceeds the maximum directory depth "
                f"of {MAX_DIR_DEPTH}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        files.append((p, rel))

    # Sort for deterministic ordering
    files.sort(key=lambda pr: pr[1].as_posix())
    return files


def _validate_files(files: list[tuple[Path, Path]]) -> None:
    """Validate files based on spec."""
    if len(files) == 0:
        typer.secho(
            "Nothing to upload: no files matched after applying .gitignore and "
            f"allowed extensions {sorted(ALLOWED_EXTS)}.",
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
    for abs_p, rel_p in files:
        size = abs_p.stat().st_size
        total += size

        # Check single file size
        if size > MAX_FILE_BYTES:
            typer.secho(
                f"File too large: '{rel_p.as_posix()}' is {size} bytes, "
                f"exceeding the per-file limit of {MAX_FILE_BYTES} bytes.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        # Ensure we can decode as UTF-8.
        try:
            abs_p.read_text(encoding=UTF8)
        except UnicodeDecodeError as err:
            typer.secho(
                f"Encoding error: '{rel_p.as_posix()}' is not UTF-8 encoded.",
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
    for abs_p, rel_p in files:
        rel_posix = rel_p.as_posix()
        mime = _detect_mime(abs_p)

        fobj = stack.enter_context(
            open(abs_p, "rb")  # pylint: disable=consider-using-with
        )
        typer.echo(f"Attach {rel_posix} ({mime}, {abs_p.stat().st_size} B)")

        form.append(("files", (rel_posix, fobj, mime)))
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


def _validate_credentials_content(creds_path: Path) -> str:
    """Load and validate the credentials file content.

    Ensures required keys exist:
      - AUTHN_TYPE_JSON_KEY
      - ACCESS_TOKEN_KEY
      - REFRESH_TOKEN_KEY
    """
    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        typer.secho(
            f"Invalid credentials file at '{creds_path}': {err}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from err

    required_keys = [AUTHN_TYPE_JSON_KEY, ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY]
    missing = [key for key in required_keys if key not in creds]

    if missing:
        typer.secho(
            f"Credentials file '{creds_path}' is missing "
            f"required key(s): {', '.join(missing)}. Please log in again.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    return creds[ACCESS_TOKEN_KEY]


def publish(
    app: Annotated[
        Path,
        typer.Argument(
            help="Project directory to upload (defaults to current directory)."
        ),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation used for login before publishing app."),
    ] = None,
    token: Annotated[
        Optional[str],
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
        token = _validate_credentials_content(creds_path)

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

    if resp.status_code == 200:
        typer.secho("🎊 Upload successful", fg=typer.colors.GREEN, bold=True)
        return  # success

    # Error path:
    msg = f"❌ Upload failed with status {resp.status_code}"
    if resp.text:
        msg += f": {resp.text}"
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)

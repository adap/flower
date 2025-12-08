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
"""Flower command line interface `new` command."""


import io
import zipfile
from pathlib import Path
from typing import Annotated

import requests
import typer

from flwr.supercore.constant import PLATFORM_API_URL

from ..utils import parse_app_spec, prompt_text, request_download_link


def print_success_prompt(package_name: str) -> None:
    """Print styled setup instructions for running a new Flower App after creation."""
    prompt = typer.style(
        "🎊 Flower App creation successful.\n\n"
        "To run your Flower App, first install its dependencies:\n\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        f"	cd {package_name} && pip install -e .\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    prompt += typer.style(
        "then, run the app:\n\n ",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        "\tflwr run .\n\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    prompt += typer.style(
        "💡 Check the README in your app directory to learn how to\n"
        "customize it and how to run it using the Deployment Runtime.\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    print(prompt)


# Security: prevent zip-slip
def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Extract ZIP file into destination directory."""
    dest_dir = dest_dir.resolve()

    def _is_within_directory(base: Path, target: Path) -> bool:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False

    for member in zf.infolist():
        # Skip directory placeholders;
        # ZipInfo can represent them as names ending with '/'.
        if member.is_dir():
            target_path = (dest_dir / member.filename).resolve()
            if not _is_within_directory(dest_dir, target_path):
                raise ValueError(f"Unsafe path in zip: {member.filename}")
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        # Files
        target_path = (dest_dir / member.filename).resolve()
        if not _is_within_directory(dest_dir, target_path):
            raise ValueError(f"Unsafe path in zip: {member.filename}")

        # Ensure parent exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract
        with zf.open(member, "r") as src, open(target_path, "wb") as dst:
            dst.write(src.read())


def _download_zip_to_memory(presigned_url: str) -> io.BytesIO:
    """Download ZIP file from Platform API to memory."""
    try:
        r = requests.get(presigned_url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        typer.secho(
            f"ZIP download failed: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e

    buf = io.BytesIO(r.content)
    # Validate it's a zip
    if not zipfile.is_zipfile(buf):
        typer.secho(
            "Downloaded file is not a valid ZIP",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    buf.seek(0)
    return buf


def download_remote_app_via_api(app_spec: str) -> None:
    """Download App from Platform API."""
    # Validate app version and ID format
    app_id, app_version = parse_app_spec(app_spec)
    app_name = app_id.split("/")[1]

    project_dir = Path.cwd() / app_name
    if project_dir.exists():
        if not typer.confirm(
            typer.style(
                f"\n💬 {app_name} already exists, do you want to override it?",
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        ):
            return

    print(
        typer.style(
            f"\n🔗 Requesting download link for {app_id}...",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    # Fetch ZIP downloading URL
    url = f"{PLATFORM_API_URL}/hub/fetch-zip"
    presigned_url = request_download_link(app_id, app_version, url, "zip_url")

    print(
        typer.style(
            "⬇️  Downloading ZIP into memory...",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    zip_buf = _download_zip_to_memory(presigned_url)

    print(
        typer.style(
            f"📦 Unpacking into {project_dir}...",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    with zipfile.ZipFile(zip_buf) as zf:
        _safe_extract_zip(zf, Path.cwd())

    print_success_prompt(app_name)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def new(
    app_name: Annotated[
        str | None,
        typer.Argument(
            help="Flower app id. Use the format "
            "'@account_name/app_name' or '@account_name/app_name==x.y.z'. "
            "Version is optional (defaults to latest)."
        ),
    ] = None,
) -> None:
    """Create new Flower App."""
    if app_name is None:
        app_name = prompt_text("Please provide the app id")

    # Download remote app
    download_remote_app_via_api(app_name)

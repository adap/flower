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
"""Flower command line interface `app review` command."""


import hashlib
import re
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import requests
import typer

from ..install import install_from_fab
from ..utils import request_download_link
from flwr.common.config import get_flwr_dir
from flwr.supercore.constant import APP_ID_PATTERN, PLATFORM_API_URL
from flwr.supercore.primitives.asymmetric_ed25519 import (
    create_signed_message,
    sign_message,
)


def _mk_review_dir(publisher: str, app_name: str) -> Path:
    """Create a directory for reviewing code."""
    home = get_flwr_dir
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    d = home / "reviews" / f"{ts}--@{publisher}--{app_name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _request_download_link(app_id: str, version: Optional[str]) -> str:
    """Request download link from Flower platform API."""
    url = f"{PLATFORM_API_URL}/hub/fetch-fab"

    return request_download_link(app_id, version, url, "fab_url")


def _download_fab(url: str) -> bytes:
    """Download FAB file from given URL."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        typer.secho(
            f"❌ FAB download failed: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e
    return r.content


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    """Load a private key (Ed25519) using cryptography."""
    try:
        pem = path.read_bytes()
    except OSError as e:
        typer.secho(f"❌ Failed to read private key: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        private_key = serialization.load_pem_private_key(pem, password=None)
    except ValueError as e:
        typer.secho(f"❌ Invalid private key format: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        typer.secho("❌ Private key is not Ed25519", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    return private_key


def _sign_fab(fab_bytes: bytes, private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Sign the given FAB hash bytes."""
    # Get current timestamp
    timestamp = int(time.time())
    signed_message = create_signed_message(
        hashlib.sha256(fab_bytes).digest(),
        timestamp,
    )
    return sign_message(private_key, signed_message)


def _submit_review(app_id: str, sig_hex: str, token: str) -> None:
    """Submit review to Flower Platform API."""
    url = f"{PLATFORM_API_URL}/hub/apps/review"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "app_id": app_id,
        "signature_hex": sig_hex,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        typer.secho(f"❌ Network error while submitting review: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if resp.status_code // 100 == 2:
        typer.secho("🎊 Review submitted", fg=typer.colors.GREEN, bold=True)
        return

    # Error path:
    msg = f"❌ Review submission failed (HTTP {resp.status_code})"
    if resp.text:
        msg += f": {resp.text}"
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def review(
    app_id: Annotated[
        str,
        typer.Argument(
            help="App identifier in the form @user_name/app_name)."
        ),
    ],
    version: Annotated[
        Optional[str],
        typer.Option(
            "--version",
            help="Version of the app to review (e.g., '1.0.0').",
        ),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            help="Bearer token for Platform API.",
        ),
    ] = None,
) -> None:
    """
    Download a FAB for <APP-ID>, unpack it for manual review, and upon confirmation
    sign & submit the review to the Platform.
    """
    if not token:
        typer.secho(
            "❌ Missing authentication token. "
            "Please run `flwr login` to generate one.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate app_id format
    m = re.match(APP_ID_PATTERN, app_id)
    if not m:
        typer.secho(
            "❌ Invalid remote app ID. Expected "
            "format: '@user_name/app_name'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    publisher, app_name = m.group(1), m.group(2)  # (publisher, app_name)

    # Download FAB
    typer.secho("Downloading FAB... ", fg=typer.colors.BLUE)
    url = _request_download_link(app_id, version)
    fab_bytes = _download_fab(url)

    # Unpack FAB
    typer.secho("Unpacking FAB... ", fg=typer.colors.BLUE)
    review_dir = _mk_review_dir(publisher, app_name)
    install_from_fab(fab_bytes, review_dir)

    # Prompt to ask for sign
    typer.secho(
        f"""
    Review the unpacked app in the following directory:

        {review_dir}

    If you have reviewed the app and want to continue to sign it,
    type {typer.style("SIGN", fg=typer.colors.GREEN, bold=True)} or abort with CTRL+C.
    """,
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    confirmation = typer.prompt("Type SIGN to continue").strip()
    if confirmation.upper() != "SIGN":
        typer.secho("Aborted (user did not type SIGN).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=130)

    # Ask for private key path
    key_path_str = typer.prompt("Please specify the path of Ed25519 private key (PEM) for signing")
    key_path = Path(key_path_str).expanduser().resolve()
    if not key_path.is_file():
        typer.secho(f"❌ Private key not found: {key_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Load private key and sign FAB
    private_key = _load_private_key(key_path)
    signature = _sign_fab(fab_bytes, private_key)
    sig_hex = signature.hex()

    # Submit review
    _submit_review(app_id=app_id, sig_hex=sig_hex, token=token)

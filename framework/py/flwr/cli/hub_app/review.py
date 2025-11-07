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


from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import zipfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, Tuple

import requests
import typer

from flwr.supercore.constant import APP_ID_PATTERN, PLATFORM_API_URL

# ──────────────────────────────────────────────────────────────────────────────
# Constants (replace with imports if you already define these centrally)
FLWR_DIR = ".flwr"

APP_ID_RE = re.compile(r"^@([A-Za-z0-9_]+)/([A-Za-z0-9_\-]+)$")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: credentials, token, home paths

def _get_flwr_home() -> Path:
    # Prefer FLWR_HOME; fallback to ~/.flwr
    env = os.getenv("FLWR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(FLWR_DIR).resolve()


def _mk_review_dir(home: Path, publisher: str, app_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    d = home / "reviews" / f"{ts}--@{publisher}--{app_name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _request_download_link(app_id: str) -> str:
    """Request download link from Flower platform API."""
    url = f"{PLATFORM_API_URL}/hub/fetch-fab"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "app_id": app_id,  # send raw string of app_id
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    except requests.RequestException as e:
        raise typer.BadParameter(f"Unable to connect to Platform API: {e}") from e

    if resp.status_code == 404:
        raise typer.BadParameter(f"'{app_id}' not found in Platform API")
    if not resp.ok:
        raise typer.BadParameter(
            f"Platform API request failed with "
            f"status {resp.status_code}. Details: {resp.text}"
        )

    data = resp.json()
    if "fab_url" not in data:
        raise typer.BadParameter("Invalid response from Platform API")
    return str(data["fab_url"])


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


def _save_fab_archive(review_dir: Path, publisher: str, app_name: str, data: bytes) -> Path:
    # Use a neutral extension; content type may vary
    archive_path = review_dir / f"@{publisher}--{app_name}.fab"
    archive_path.write_bytes(data)
    return archive_path


def _unpack_archive(archive_path: Path, dest_dir: Path) -> None:
    # Try ZIP first
    try:
        with zipfile.ZipFile(io.BytesIO(archive_path.read_bytes())) as zf:
            zf.extractall(dest_dir)
            return
    except zipfile.BadZipFile:
        pass

    # Try TAR (supports tar, tar.gz, tar.bz2, tar.xz)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_path.read_bytes())) as tf:
            tf.extractall(dest_dir)
            return
    except tarfile.TarError:
        pass

    typer.secho(
        f"Unrecognized archive format for '{archive_path.name}'. "
        "Expected a ZIP or TAR-based FAB archive.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: signing

def _sha256_digest_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()  # NOTE: digest() returns raw bytes


def _load_private_key(path: Path):
    """Load a private key (supports RSA and Ed25519) using cryptography."""
    try:
        from cryptography.hazmat.primitives import serialization
    except Exception as e:  # pragma: no cover
        typer.secho(
            f"Missing dependency 'cryptography' for signing: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        pem = path.read_bytes()
    except OSError as e:
        typer.secho(f"Failed to read private key: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    password: Optional[bytes] = None  # adapt if you want passphrase prompts
    try:
        key = serialization.load_pem_private_key(pem, password=password)
    except ValueError as e:
        typer.secho(f"Invalid private key format: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return key


def _sign_fab_hash(fab_hash_bytes: bytes, key) -> bytes:
    """Sign the given FAB hash bytes. Supports RSA (PKCS#1 v1.5 + SHA256) and Ed25519.

    For Ed25519, we sign the 32-byte hash value directly as the message.
    For RSA, we sign the hash *as a message* using PKCS#1 v1.5 with SHA256.
    """
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, rsa, ed25519
    except Exception as e:  # pragma: no cover
        typer.secho(
            f"Missing dependency 'cryptography' for signing: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key.sign(fab_hash_bytes)

    if isinstance(key, rsa.RSAPrivateKey):
        return key.sign(
            fab_hash_bytes,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )

    typer.secho(
        "Unsupported private key type. Only RSA and Ed25519 are supported.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)

# ──────────────────────────────────────────────────────────────────────────────
# Submit review

def _submit_review(app_id: str, fab_hash_hex: str, sig_hex: str, token: str) -> None:
    url = f"{PLATFORM_API_URL}/hub/apps/review"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "app_id": app_id,
        "fab_hash_hex": fab_hash_hex,
        "signature_hex": sig_hex,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        typer.secho(f"Network error while submitting review: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if resp.status_code // 100 == 2:
        typer.secho("✓ Review submitted", fg=typer.colors.GREEN, bold=True)
        # Print any JSON response for convenience
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            try:
                typer.echo(json.dumps(resp.json(), indent=2))
            except Exception:
                if resp.text:
                    typer.echo(resp.text)
        else:
            if resp.text:
                typer.echo(resp.text)
        return

    msg = f"Review submission failed (HTTP {resp.status_code})"
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
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            help="Bearer token for Platform API.",
        ),
    ] = None,
):
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
    url = _request_download_link(app_id)
    fab_bytes = _download_fab(url)

    home = _get_flwr_home()
    review_dir = _mk_review_dir(home, publisher, app_name)
    archive_path = _save_fab_archive(review_dir, publisher, app_name, fab_bytes)

    typer.echo("Unpacking FAB...")
    unpack_dir = review_dir / "unpacked"
    unpack_dir.mkdir(parents=True, exist_ok=False)
    _unpack_archive(archive_path, unpack_dir)

    # 2) Review Preparation
    typer.echo()
    typer.echo("Review the unpacked app in the following directory:\n")
    typer.echo(f"    {unpack_dir}")
    typer.echo()
    typer.echo("If you have reviewed the app and want to continue to sign it type SIGN or abort with CTRL+C")
    confirmation = typer.prompt("Type SIGN to continue").strip()
    if confirmation.upper() != "SIGN":
        typer.secho("Aborted (user did not type SIGN).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=130)

    # 3) Signing & Submission
    # Ask for private key path
    key_path_str = typer.prompt("Please specify the private key for signing")
    key_path = Path(key_path_str).expanduser().resolve()
    if not key_path.is_file():
        typer.secho(f"Private key not found: {key_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    key = _load_private_key(key_path)
    fab_hash_bytes = _sha256_digest_bytes(fab_bytes)  # NOTE: digest() returns raw bytes (not hex)
    signature = _sign_fab_hash(fab_hash_bytes, key)

    fab_hash_hex = fab_hash_bytes.hex()
    sig_hex = signature.hex()

    _submit_review(app_id=app_id, fab_hash_hex=fab_hash_hex, sig_hex=sig_hex, token=token)

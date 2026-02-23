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
"""Helpers for lazily managing a local SuperLink runtime from the CLI."""


from __future__ import annotations

import subprocess
import time
from pathlib import Path

import click
import grpc
import typer

from flwr.common.constant import ISOLATION_MODE_SUBPROCESS
from flwr.common.grpc import create_channel
from flwr.supercore.utils import get_flwr_home

from .constant import (
    CONTROL_API_PROBE_INTERVAL,
    CONTROL_API_PROBE_TIMEOUT,
    LOCAL_CONTROL_API_ADDRESS,
    LOCAL_SIMULATIONIO_API_ADDRESS,
    LOCAL_SUPERLINK_STARTUP_TIMEOUT,
)
from .typing import SuperLinkConnection


def ensure_local_superlink(connection: SuperLinkConnection) -> SuperLinkConnection:
    """Ensure local SuperLink availability for local simulation connections.

    If the provided connection represents a local simulation configuration without an
    explicit address, this helper lazily starts a managed local SuperLink (simulation
    mode) when no Control API endpoint is available.

    Connections with an explicit address are treated as user-managed and returned
    unchanged.
    """
    if connection.options is None:
        return connection

    # Options-only local profile (for example: [superlink.local] with options.* only).
    if connection.address is None:
        runtime_connection = SuperLinkConnection(
            name=connection.name,
            address=LOCAL_CONTROL_API_ADDRESS,
            root_certificates=None,
            insecure=True,
            federation=connection.federation,
            options=connection.options,
        )
        if not _is_local_superlink_started():
            _start_local_superlink()
        return runtime_connection

    # Explicit addresses are user-managed.
    return connection


def _get_local_superlink_paths() -> tuple[Path, Path, Path]:
    """Return (database_path, storage_dir, log_file_path) for local SuperLink."""
    runtime_dir = get_flwr_home() / "local-superlink"
    database_path = runtime_dir / "state.db"
    storage_dir = runtime_dir / "ffs"
    log_file_path = runtime_dir / "superlink.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return database_path, storage_dir, log_file_path


def _is_local_superlink_started() -> bool:
    """Return True if local SuperLink's Control API endpoint is reachable."""
    channel = create_channel(server_address=LOCAL_CONTROL_API_ADDRESS, insecure=True)
    try:
        grpc.channel_ready_future(channel).result(timeout=CONTROL_API_PROBE_TIMEOUT)
        return True
    except (grpc.FutureTimeoutError, grpc.RpcError):
        return False
    finally:
        channel.close()


def _start_local_superlink() -> None:
    """Start a managed local SuperLink in simulation mode and wait for readiness."""
    database_path, storage_dir, log_file_path = _get_local_superlink_paths()

    typer.secho(
        f"Starting local SuperLink on {LOCAL_CONTROL_API_ADDRESS}...",
        fg=typer.colors.BLUE,
    )

    command = [
        "flower-superlink",
        "--insecure",
        "--simulation",
        "--isolation",
        ISOLATION_MODE_SUBPROCESS,
        "--control-api-address",
        LOCAL_CONTROL_API_ADDRESS,
        "--simulationio-api-address",
        LOCAL_SIMULATIONIO_API_ADDRESS,
        "--database",
        str(database_path),
        "--storage-dir",
        str(storage_dir),
    ]

    # Keep process detached and route stdout/stderr to a persistent log file.
    try:
        with log_file_path.open("ab") as log_file:
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        raise click.ClickException(
            f"Unable to launch `flower-superlink` for local simulation: {exc}"
        ) from exc

    deadline = time.monotonic() + LOCAL_SUPERLINK_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        # Early exit when local SuperLink process has terminated
        if process.poll() is not None:
            raise click.ClickException(
                "Failed to start local SuperLink: "
                f"`flower-superlink` exited with code {process.poll()}. "
                f"See logs at {log_file_path}."
            )

        if _is_local_superlink_started():
            return
        time.sleep(CONTROL_API_PROBE_INTERVAL)

    # Timeout while waiting for local SuperLink to start
    raise click.ClickException(
        "Failed to start local SuperLink within "
        f"{LOCAL_SUPERLINK_STARTUP_TIMEOUT:.0f}s. "
        f"See logs at {log_file_path}."
    )

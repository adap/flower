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

import hashlib
import subprocess
import time
from pathlib import Path
from typing import cast

import click
import grpc
import typer

from flwr.common.constant import (
    CONTROL_API_PORT,
    ISOLATION_MODE_SUBPROCESS,
    SIMULATIONIO_PORT,
)
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.supercore.address import parse_address, resolve_bind_address
from flwr.supercore.utils import get_flwr_home

from .config_utils import load_certificate_in_connection
from .typing import SuperLinkConnection

DEFAULT_LOCAL_CONTROL_API_ADDRESS = f"127.0.0.1:{CONTROL_API_PORT}"
LOCAL_SUPERLINK_STARTUP_TIMEOUT_SEC = 15.0
CONTROL_API_PROBE_TIMEOUT_SEC = 0.4
CONTROL_API_PROBE_INTERVAL_SEC = 0.2


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
        runtime_connection = _runtime_connection_from(connection)
        if not _is_control_api_available(runtime_connection):
            _start_local_superlink(runtime_connection)
        return runtime_connection

    # Explicit addresses are user-managed.
    return connection


def _runtime_connection_from(connection: SuperLinkConnection) -> SuperLinkConnection:
    """Return an effective connection for managed local runtime."""
    raw_address = connection.address or DEFAULT_LOCAL_CONTROL_API_ADDRESS
    address = resolve_bind_address(raw_address)
    return SuperLinkConnection(
        name=connection.name,
        address=address,
        root_certificates=None,
        insecure=True,
        federation=connection.federation,
        options=connection.options,
    )


def _format_host_port(host: str, port: int, is_v6: bool | None) -> str:
    """Build host:port, preserving IPv6 bracket format."""
    return f"[{host}]:{port}" if is_v6 else f"{host}:{port}"


def _derive_simulationio_address(control_api_address: str) -> str:
    """Derive the SimulationIo API address from the Control API address."""
    parsed = parse_address(control_api_address)
    if not parsed:
        return f"127.0.0.1:{SIMULATIONIO_PORT}"

    host, _, is_v6 = parsed
    return _format_host_port(host, int(SIMULATIONIO_PORT), is_v6)


def _runtime_paths_for_address(address: str) -> tuple[Path, Path, Path]:
    """Return (database_path, storage_dir, log_file_path) for managed local runtime."""
    digest = hashlib.sha256(address.encode("utf-8")).hexdigest()[:16]
    runtime_dir = get_flwr_home() / "superlink" / digest
    database_path = runtime_dir / "state.db"
    storage_dir = runtime_dir / "ffs"
    log_file_path = runtime_dir / "superlink.log"
    return database_path, storage_dir, log_file_path


def _is_control_api_available(
    connection: SuperLinkConnection,
    probe_timeout_sec: float = CONTROL_API_PROBE_TIMEOUT_SEC,
) -> bool:
    """Return True if the connection's Control API endpoint is reachable."""
    if connection.address is None:
        return False

    root_certificates = load_certificate_in_connection(connection)
    channel = create_channel(
        server_address=connection.address,
        insecure=connection.insecure,
        root_certificates=root_certificates,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=probe_timeout_sec)
        return True
    except (grpc.FutureTimeoutError, grpc.RpcError):
        return False
    finally:
        channel.close()


def _start_local_superlink(runtime_connection: SuperLinkConnection) -> None:
    """Start a managed local SuperLink in simulation mode and wait for readiness."""
    address = cast(str, runtime_connection.address)
    simulationio_address = _derive_simulationio_address(address)
    database_path, storage_dir, log_file_path = _runtime_paths_for_address(address)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"Starting local SuperLink on {address}...",
        fg=typer.colors.BLUE,
    )

    command = [
        "flower-superlink",
        "--insecure",
        "--simulation",
        "--isolation",
        ISOLATION_MODE_SUBPROCESS,
        "--control-api-address",
        address,
        "--simulationio-api-address",
        simulationio_address,
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
            "Unable to launch `flower-superlink`. Ensure Flower is installed in the "
            "active environment."
        ) from exc

    deadline = time.monotonic() + LOCAL_SUPERLINK_STARTUP_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if _is_control_api_available(runtime_connection):
            return
        time.sleep(CONTROL_API_PROBE_INTERVAL_SEC)

    process_state = process.poll()
    details = (
        f" Process exited with code {process_state}."
        if process_state is not None
        else ""
    )
    raise click.ClickException(
        "Failed to start local SuperLink within "
        f"{LOCAL_SUPERLINK_STARTUP_TIMEOUT_SEC:.0f}s.{details} "
        f"See logs at {log_file_path}."
    )

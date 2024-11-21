# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `log` command."""

import time
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from typing import Annotated, Any, Optional, cast

import grpc
import typer

from flwr.cli.config_utils import (
    load_and_validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_project_config,
)
from flwr.common.constant import CONN_RECONNECT_INTERVAL, CONN_REFRESH_PERIOD
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log as logger
from flwr.proto.exec_pb2 import StreamLogsRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


def start_stream(
    run_id: int, channel: grpc.Channel, refresh_period: int = CONN_REFRESH_PERIOD
) -> None:
    """Start log streaming for a given run ID."""
    stub = ExecStub(channel)
    after_timestamp = 0.0
    try:
        logger(INFO, "Starting logstream for run_id `%s`", run_id)
        while True:
            after_timestamp = stream_logs(run_id, stub, refresh_period, after_timestamp)
            time.sleep(CONN_RECONNECT_INTERVAL)
            logger(DEBUG, "Reconnecting to logstream")
    except KeyboardInterrupt:
        logger(INFO, "Exiting logstream")
    except grpc.RpcError as e:
        # pylint: disable=E1101
        if e.code() == grpc.StatusCode.NOT_FOUND:
            logger(ERROR, "Invalid run_id `%s`, exiting", run_id)
        if e.code() == grpc.StatusCode.CANCELLED:
            pass
    finally:
        channel.close()


def stream_logs(
    run_id: int, stub: ExecStub, duration: int, after_timestamp: float
) -> float:
    """Stream logs from the beginning of a run with connection refresh.

    Parameters
    ----------
    run_id : int
        The identifier of the run.
    stub : ExecStub
        The gRPC stub to interact with the Exec service.
    duration : int
        The timeout duration for each stream connection in seconds.
    after_timestamp : float
        The timestamp to start streaming logs from.

    Returns
    -------
    float
        The latest timestamp from the streamed logs or the provided `after_timestamp`
        if no logs are returned.
    """
    req = StreamLogsRequest(run_id=run_id, after_timestamp=after_timestamp)

    latest_timestamp = 0.0
    res = None
    try:
        for res in stub.StreamLogs(req, timeout=duration):
            print(res.log_output, end="")
    except grpc.RpcError as e:
        # pylint: disable=E1101
        if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
            raise e
    finally:
        if res is not None:
            latest_timestamp = cast(float, res.latest_timestamp)

    return max(latest_timestamp, after_timestamp)


def print_logs(run_id: int, channel: grpc.Channel, timeout: int) -> None:
    """Print logs from the beginning of a run."""
    stub = ExecStub(channel)
    req = StreamLogsRequest(run_id=run_id)

    try:
        while True:
            try:
                # Enforce timeout for graceful exit
                for res in stub.StreamLogs(req, timeout=timeout):
                    print(res.log_output)
            except grpc.RpcError as e:
                # pylint: disable=E1101
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    break
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    logger(ERROR, "Invalid run_id `%s`, exiting", run_id)
                    break
                if e.code() == grpc.StatusCode.CANCELLED:
                    break
    except KeyboardInterrupt:
        logger(DEBUG, "Stream interrupted by user")
    finally:
        channel.close()
        logger(DEBUG, "Channel closed")


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    logger(DEBUG, channel_connectivity)


def log(
    run_id: Annotated[
        int,
        typer.Argument(help="The Flower run ID to query"),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project to run"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to run the app on"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--show",
            help="Flag to stream or print logs from the Flower run",
        ),
    ] = True,
) -> None:
    """Get logs from a Flower project run."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / "pyproject.toml" if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" not in federation_config:
        typer.secho(
            "❌ `flwr log` currently works with Exec API. Ensure that the correct"
            "Exec API address is provided in the `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    _log_with_exec_api(app, federation_config, run_id, stream)


def _log_with_exec_api(
    app: Path,
    federation_config: dict[str, Any],
    run_id: int,
    stream: bool,
) -> None:

    insecure, root_certificates_bytes = validate_certificate_in_federation_config(
        app, federation_config
    )
    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)

    if stream:
        start_stream(run_id, channel, CONN_REFRESH_PERIOD)
    else:
        logger(INFO, "Printing logstream for run_id `%s`", run_id)
        print_logs(run_id, channel, timeout=5)

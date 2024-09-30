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

import sys
import time
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from typing import Annotated, Optional

import grpc
import typer

from flwr.cli.config_utils import load_and_validate
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log as logger
from flwr.proto.exec_pb2 import StreamLogsRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


def start_stream(
    run_id: int, channel: grpc.Channel, refresh_period: int = CONN_REFRESH_PERIOD
) -> None:
    """Start log streaming for a given run ID."""
    try:
        while True:
            logger(INFO, "Starting logstream for run_id `%s`", run_id)
            stream_logs(run_id, channel, refresh_period)
            time.sleep(2)
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


def stream_logs(run_id: int, channel: grpc.Channel, duration: int) -> None:
    """Stream logs from the beginning of a run with connection refresh."""
    start_time = time.time()
    stub = ExecStub(channel)
    req = StreamLogsRequest(run_id=run_id)

    for res in stub.StreamLogs(req):
        print(res.log_output)
        if time.time() - start_time > duration:
            break


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

    if config is None:
        typer.secho(
            "Project configuration could not be loaded.\n"
            "pyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        sys.exit()

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    typer.secho("Success", fg=typer.colors.GREEN)

    federation = federation or config["tool"]["flwr"]["federations"].get("default")

    if federation is None:
        typer.secho(
            "❌ No federation name was provided and the project's `pyproject.toml` "
            "doesn't declare a default federation (with a SuperExec address or an "
            "`options.num-supernodes` value).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Validate the federation exists in the configuration
    federation_config = config["tool"]["flwr"]["federations"].get(federation)
    if federation_config is None:
        available_feds = {
            fed for fed in config["tool"]["flwr"]["federations"] if fed != "default"
        }
        typer.secho(
            f"❌ There is no `{federation}` federation declared in the "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if "address" not in federation_config:
        typer.secho(
            "❌ `flwr log` currently works with `SuperExec`. Ensure that the correct"
            "`SuperExec` address is provided in the `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    _log_with_superexec(federation_config, run_id, stream)


# pylint: disable-next=too-many-branches
def _log_with_superexec(
    federation_config: dict[str, str],
    run_id: int,
    stream: bool,
) -> None:
    insecure_str = federation_config.get("insecure")
    if root_certificates := federation_config.get("root-certificates"):
        root_certificates_bytes = Path(root_certificates).read_bytes()
        if insecure := bool(insecure_str):
            typer.secho(
                "❌ `root_certificates` were provided but the `insecure` parameter"
                "is set to `True`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
    else:
        root_certificates_bytes = None
        if insecure_str is None:
            typer.secho(
                "❌ To disable TLS, set `insecure = true` in `pyproject.toml`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)
        if not (insecure := bool(insecure_str)):
            typer.secho(
                "❌ No certificate were given yet `insecure` is set to `False`.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

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

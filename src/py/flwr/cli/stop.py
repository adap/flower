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
"""Flower command line interface `stop` command."""


from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Optional

import grpc
import typer

from flwr.cli.config_utils import (
    load_and_validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_project_config,
)
from flwr.common.constant import FAB_CONFIG_FILE
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StopRunRequest, StopRunResponse  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


def stop(
    run_id: Annotated[  # pylint: disable=unused-argument
        int,
        typer.Argument(help="The Flower run ID to stop"),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation"),
    ] = None,
) -> None:
    """Stop a run."""
    # Load and validate federation config
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / FAB_CONFIG_FILE if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" not in federation_config:
        typer.secho(
            "❌ `flwr stop` currently works with Exec API. Ensure that the correct"
            "Exec API address is provided in the `pyproject.toml`.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    try:
        channel = _init_channel(app, federation_config)
        stub = ExecStub(channel)  # pylint: disable=unused-variable # noqa: F841

        typer.secho(f"✋ Stopping run ID {run_id}...", fg=typer.colors.GREEN)
        _stop_run(stub, run_id=run_id)

    except ValueError as err:
        typer.secho(
            f"❌ {err}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err
    finally:
        channel.close()


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


def _init_channel(app: Path, federation_config: dict[str, Any]) -> grpc.Channel:
    """Initialize gRPC channel to the Exec API."""
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
    return channel


def _stop_run(
    stub: ExecStub,  # pylint: disable=unused-argument
    run_id: int,  # pylint: disable=unused-argument
) -> None:
    """Stop a run."""
    response: StopRunResponse = stub.StopRun(request=StopRunRequest(run_id=run_id))

    if response.success:
        typer.secho(f"✅ Run {run_id} successfully stopped.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"❌ Run {run_id} couldn't be stopped.", fg=typer.colors.RED)

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
"""Flower command line interface `pull` command."""


from pathlib import Path
from typing import Annotated

import typer

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.constant import FAB_CONFIG_FILE
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    PullArtifactsRequest,
    PullArtifactsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from .utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def pull(  # pylint: disable=R0914
    run_id: Annotated[
        int,
        typer.Option(
            "--run-id",
            help="Run ID to pull artifacts from.",
        ),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation."),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
) -> None:
    """Pull artifacts from a Flower run."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / FAB_CONFIG_FILE if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = process_loaded_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config, federation_config_overrides
    )
    exit_if_no_address(federation_config, "pull")
    channel = None
    try:

        auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
        channel = init_channel(app, federation_config, auth_plugin)
        stub = ControlStub(channel)
        with flwr_cli_grpc_exc_handler():
            res: PullArtifactsResponse = stub.PullArtifacts(
                PullArtifactsRequest(run_id=run_id)
            )

        if not res.url:
            typer.secho(
                f"❌ A download URL for artifacts from run {run_id} couldn't be "
                "obtained.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1)

        typer.secho(
            f"✅ Artifacts for run {run_id} can be downloaded from: {res.url}",
            fg=typer.colors.GREEN,
        )
    finally:
        if channel:
            channel.close()

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
"""Flower command line interface `run` command."""

import sys
from logging import DEBUG
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from flwr.cli import config_utils
from flwr.cli.build import build
from flwr.common.config import parse_config_args
from flwr.common.constant import SUPEREXEC_DEFAULT_ADDRESS
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


# pylint: disable-next=too-many-locals
def run(
    directory: Annotated[
        Optional[Path],
        typer.Option(help="Path of the Flower project to run"),
    ] = None,
    config_overrides: Annotated[
        Optional[str],
        typer.Option(
            "--run-config",
            "-c",
            help="Override configuration key-value pairs",
        ),
    ] = None,
) -> None:
    """Run Flower project."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = directory / "pyproject.toml" if directory else None
    config, errors, warnings = config_utils.load_and_validate(path=pyproject_path)

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

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        log(DEBUG, channel_connectivity)

    channel = create_channel(
        server_address=SUPEREXEC_DEFAULT_ADDRESS,
        insecure=True,
        root_certificates=None,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)
    stub = ExecStub(channel)

    fab_path = build(directory)

    req = StartRunRequest(
        fab_file=Path(fab_path).read_bytes(),
        override_config=parse_config_args(config_overrides, separator=","),
    )
    res = stub.StartRun(req)
    typer.secho(f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN)

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

from flwr.cli.build import build
from flwr.cli.config_utils import load_and_validate
from flwr.common.config import parse_config_args
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub
from flwr.simulation.run_simulation import _run_simulation


# pylint: disable-next=too-many-locals
def run(
    address: Annotated[
        Optional[str],
        typer.Option(help="The address of the SuperExec to run on"),
    ] = None,
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

    if address:

        def on_channel_state_change(channel_connectivity: str) -> None:
            """Log channel connectivity."""
            log(DEBUG, channel_connectivity)

        channel = create_channel(
            server_address=address,
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

    else:
        server_app_ref = config["flower"]["components"]["serverapp"]
        client_app_ref = config["flower"]["components"]["clientapp"]

        try:
            num_supernodes = config["flower"]["federations"]["localhost"]["options"][
                "num-supernodes"
            ]
        except KeyError as err:
            typer.secho(
                "❌ The project's `pyproject.toml` needs to declare "
                "`flower.federations.localhost.options.num-supernodes` if a "
                "SuperExec address is not provided.",
                fg=typer.colors.RED,
                bold=True,
            )
            raise typer.Exit(code=1) from err

        _run_simulation(
            server_app_attr=server_app_ref,
            client_app_attr=client_app_ref,
            num_supernodes=num_supernodes,
        )

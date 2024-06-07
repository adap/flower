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

from io import BytesIO
import sys
from enum import Enum
from logging import DEBUG
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from flwr.cli import config_utils
from flwr.cli.build import build
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub
from flwr.simulation.run_simulation import _run_simulation


class Engine(str, Enum):
    """Enum defining the engine to run on."""

    SIMULATION = "simulation"


# pylint: disable-next=too-many-locals
def run(
    engine: Annotated[
        Optional[Engine],
        typer.Option(case_sensitive=False, help="The ML framework to use"),
    ] = None,
    use_superexec: Annotated[
        bool,
        typer.Option(
            case_sensitive=False, help="Use this flag to use the new SuperExec API"
        ),
    ] = False,
    app_path: Annotated[
        Optional[Path],
        typer.Option(
            case_sensitive=False, help="Use this flag to use the new SuperExec API"
        ),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option(
            case_sensitive=False, help="Use this flag to stream logs"
        ),
    ] = False,

) -> None:
    """Run Flower project."""
    if use_superexec:

        def on_channel_state_change(channel_connectivity: str) -> None:
            """Log channel connectivity."""
            log(DEBUG, channel_connectivity)

        channel = create_channel(
            server_address="127.0.0.1:9093",
            insecure=True,
            root_certificates=None,
            max_message_length=GRPC_MAX_MESSAGE_LENGTH,
            interceptors=None,
        )
        channel.subscribe(on_channel_state_change)
        stub = ExecStub(channel)

        fab_path = build(app_path)

        req = StartRunRequest(fab_file=open(fab_path, "rb").read())
        res = stub.StartRun(req)
        print(res)

        if follow:
            req = FetchLogsRequest(run_id=res.run_id)
            for res in stub.FetchLogs(req):
                print(res.log_output)
    else:
        typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

        config, errors, warnings = config_utils.load_and_validate()

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
                "recommended properties:\n"
                + "\n".join([f"- {line}" for line in warnings]),
                fg=typer.colors.RED,
                bold=True,
            )

        typer.secho("Success", fg=typer.colors.GREEN)

        server_app_ref = config["flower"]["components"]["serverapp"]
        client_app_ref = config["flower"]["components"]["clientapp"]

        if engine is None:
            engine = config["flower"]["engine"]["name"]

        if engine == Engine.SIMULATION:
            num_supernodes = config["flower"]["engine"]["simulation"]["supernode"][
                "num"
            ]

            typer.secho("Starting run... ", fg=typer.colors.BLUE)
            _run_simulation(
                server_app_attr=server_app_ref,
                client_app_attr=client_app_ref,
                num_supernodes=num_supernodes,
            )
        else:
            typer.secho(
                f"Engine '{engine}' is not yet supported in `flwr run`",
                fg=typer.colors.RED,
                bold=True,
            )

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
import time
from enum import Enum
from logging import DEBUG, INFO
from pathlib import Path
from typing import Optional

import grpc
import typer
from typing_extensions import Annotated

from flwr.cli import config_utils
from flwr.cli.build import build
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub
from flwr.simulation.run_simulation import _run_simulation

from ..log import stream_logs


class Engine(str, Enum):
    """Enum defining the engine to run on."""

    SIMULATION = "simulation"


# pylint: disable-next=too-many-locals
def run(
    engine: Annotated[
        Optional[Engine],
        typer.Option(case_sensitive=False, help="The execution engine to run the app"),
    ] = None,
    use_superexec: Annotated[
        bool,
        typer.Option(
            case_sensitive=False, help="Use this flag to use the new SuperExec API"
        ),
    ] = False,
    superexec_address: Annotated[
        Optional[str],
        typer.Option(case_sensitive=False, help="The address of the SuperExec server"),
    ] = "localhost:9093",
    app_path: Annotated[
        Optional[Path],
        typer.Option(
            case_sensitive=False, help="Path to a directory created with `flwr new`"
        ),
    ] = None,
    fab_path: Annotated[
        Optional[Path],
        typer.Option(case_sensitive=False, help="Path to a FAB."),
    ] = None,
    period: Annotated[
        int,
        typer.Option(
            case_sensitive=False,
            help="Use this to set connection refresh time period (in seconds)",
        ),
    ] = 60,
    follow: Annotated[
        bool,
        typer.Option(case_sensitive=False, help="Use this flag to stream logs"),
    ] = True,
    root_certificates: Annotated[
        Optional[Path],
        typer.Option(
            case_sensitive=False,
            help="Specifies the path to the PEM-encoded root certificate file for "
            "establishing secure HTTPS connections..",
        ),
    ] = None,
) -> None:
    """Run Flower project."""
    if use_superexec:

        if superexec_address is None:
            gloabl_config = config_utils.load(
                config_utils.get_flower_home() / "config.toml"
            )
            if gloabl_config:
                superexec_address = gloabl_config["federation"]["default"]
            else:
                typer.secho(
                    "No SuperExec address was provided and no global config "
                    "was found.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                sys.exit()

        assert superexec_address is not None

        def on_channel_state_change(channel_connectivity: str) -> None:
            """Log channel connectivity."""
            log(DEBUG, channel_connectivity)

        # Obtain certificates
        root_certificates = None
        if isinstance(root_certificates, str):
            root_certificates = Path(root_certificates).read_bytes()

        channel = create_channel(
            server_address=superexec_address,
            insecure=root_certificates is None,
            root_certificates=root_certificates,
            max_message_length=GRPC_MAX_MESSAGE_LENGTH,
            interceptors=None,
        )
        channel.subscribe(on_channel_state_change)
        stub = ExecStub(channel)

        if fab_path is None:
            fab_path = build(app_path, use_superexec)
        else:
            log(INFO, "Passed FAB directory, skipping `flwr build`.")

        with open(fab_path, "rb") as f:
            start_run_req = StartRunRequest(fab_file=f.read())
        start_run_res = stub.StartRun(start_run_req)
        log(INFO, "Starting run with id: %s", start_run_res.run_id)

        if follow:
            try:
                while True:
                    log(INFO, "Streaming logs")
                    stream_logs(start_run_res.run_id, channel, period)
                    time.sleep(2)
                    log(INFO, "Reconnecting to logstream")
            except KeyboardInterrupt:
                log(INFO, "Exiting logstream")
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.CANCELLED:
                    pass
            finally:
                channel.close()
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

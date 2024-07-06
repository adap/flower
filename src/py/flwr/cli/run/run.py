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

from logging import DEBUG
from pathlib import Path
from typing import Optional

import tomli
import typer
from typing_extensions import Annotated

from flwr.cli.build import build
from flwr.common.config import get_flwr_dir
from flwr.common.constant import SUPEREXEC_DEFAULT_ADDRESS
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


# pylint: disable-next=too-many-locals
def run(
    directory: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project to run"),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to run FL on"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            case_sensitive=False, help="Use this flag to print logs at `DEBUG` level"
        ),
    ] = False,
    flwr_dir: Annotated[
        Optional[str],
        typer.Argument(
            help="""Path of the Flower root.

                    By default ``flwr-dir`` is equal to:

                    - ``$FLWR_HOME/`` if ``$FLWR_HOME`` is defined
                    - ``$XDG_DATA_HOME/.flwr/`` if ``$XDG_DATA_HOME`` is defined
                    - ``$HOME/.flwr/`` in all other cases"""
        ),
    ] = None,
) -> None:
    """Run Flower project."""
    global_config_path = get_flwr_dir(flwr_dir) / "config.toml"

    if global_config_path.exists():
        with global_config_path.open("rb") as global_config_file:
            global_config = tomli.load(global_config_file)

        if federation is None:
            superexec_address = global_config["federations"][
                global_config["federation"]["default"]
            ]["address"]
        else:
            if federation in global_config["federations"]:
                superexec_address = global_config["federations"][federation]["address"]
            else:
                typer.secho(
                    f"❌ {federation} is not defined in {str(global_config_path)}.",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(code=1)
    else:
        superexec_address = SUPEREXEC_DEFAULT_ADDRESS

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        log(DEBUG, channel_connectivity)

    channel = create_channel(
        server_address=superexec_address,
        insecure=True,
        root_certificates=None,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)
    stub = ExecStub(channel)

    fab_path = build(directory)

    req = StartRunRequest(fab_file=Path(fab_path).read_bytes(), verbose=verbose)
    res = stub.StartRun(req)
    typer.secho(f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN)

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
from typing import Optional, Dict

import grpc
import typer
from typing_extensions import Annotated

from flwr.cli import config_utils
from flwr.cli.config_utils import load_and_validate
from flwr.common.config import get_flwr_dir
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log as logger

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


# pylint: disable=unused-argument
def stream_logs(run_id: int, channel: grpc.Channel, period: int) -> None:
    """Stream logs from the beginning of a run with connection refresh."""


# pylint: disable=unused-argument
def print_logs(run_id: int, channel: grpc.Channel, timeout: int) -> None:
    """Print logs from the beginning of a run."""


def log(
    run_id: Annotated[
        int,
        typer.Argument(help="The Flower run ID to query"),
    ] = None,
    directory: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project to run"),
    ] = Path("."),
    federation_name: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to run the app on"),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option(case_sensitive=False, help="Use this flag to follow logstream"),
    ] = True,
) -> None:
    """Get logs from Flower run."""
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

    federation_name = federation_name or config["tool"]["flwr"]["federations"].get(
        "default"
    )

    if federation_name is None:
        typer.secho(
            "❌ No federation name was provided and the project's `pyproject.toml` "
            "doesn't declare a default federation (with a SuperExec address or an "
            "`options.num-supernodes` value).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    # Validate the federation exists in the configuration
    federation = config["tool"]["flwr"]["federations"].get(federation_name)
    if federation is None:
        available_feds = {
            fed for fed in config["tool"]["flwr"]["federations"] if fed != "default"
        }
        typer.secho(
            f"❌ There is no `{federation_name}` federation declared in the "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if "address" in federation:
        _log_with_superexec(federation, directory)
    else:
        pass


def _log_with_superexec(
    federation: Dict[str, str],
    directory: Optional[Path],
) -> None:

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        logger(DEBUG, channel_connectivity)

    if superexec_address is None:
        global_config = config_utils.load(get_flwr_dir() / "config.toml")
        if global_config:
            superexec_address = global_config["federation"]["default"]
        else:
            typer.secho(
                "No SuperExec address was provided and no global config was found.",
                fg=typer.colors.RED,
                bold=True,
            )
            sys.exit()

    assert superexec_address is not None

    channel = create_channel(
        server_address=superexec_address,
        insecure=True,
        root_certificates=None,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=None,
    )
    channel.subscribe(on_channel_state_change)

    if follow:
        try:
            while True:
                logger(INFO, "Starting logstream for run_id `%s`", run_id)
                stream_logs(run_id, channel, CONN_REFRESH_PERIOD)
                time.sleep(2)
                logger(INFO, "Reconnecting to logstream")
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
    else:
        logger(INFO, "Printing logstream for run_id `%s`", run_id)
        print_logs(run_id, channel, timeout=5)

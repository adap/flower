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

import json
import subprocess
from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from flwr.cli.build import build
from flwr.cli.config_utils import (
    load_and_validate,
    validate_certificate_in_federation_config,
    validate_federation_in_project_config,
    validate_project_config,
)
from flwr.common.config import flatten_dict, parse_config_args
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub

from ..log import start_stream

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


# pylint: disable-next=too-many-locals
def run(
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower App to run."),
    ] = Path("."),
    federation: Annotated[
        Optional[str],
        typer.Argument(help="Name of the federation to run the app on."),
    ] = None,
    config_overrides: Annotated[
        Optional[list[str]],
        typer.Option(
            "--run-config",
            "-c",
            help="Override configuration key-value pairs, should be of the format:\n\n"
            '`--run-config \'key1="value1" key2="value2"\' '
            "--run-config 'key3=\"value3\"'`\n\n"
            "Note that `key1`, `key2`, and `key3` in this example need to exist "
            "inside the `pyproject.toml` in order to be properly overriden.",
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Use `--stream` with `flwr run` to display logs;\n "
            "logs are not streamed by default.",
        ),
    ] = False,
) -> None:
    """Run Flower App."""
    typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)

    pyproject_path = app / "pyproject.toml" if app else None
    config, errors, warnings = load_and_validate(path=pyproject_path)
    config = validate_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    if "address" in federation_config:
        _run_with_exec_api(app, federation_config, config_overrides, stream)
    else:
        _run_without_exec_api(app, federation_config, config_overrides, federation)


def _run_with_exec_api(
    app: Path,
    federation_config: dict[str, Any],
    config_overrides: Optional[list[str]],
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
    stub = ExecStub(channel)

    fab_path, fab_hash = build(app)
    content = Path(fab_path).read_bytes()
    fab = Fab(fab_hash, content)

    req = StartRunRequest(
        fab=fab_to_proto(fab),
        override_config=user_config_to_proto(parse_config_args(config_overrides)),
        federation_config=user_config_to_proto(
            flatten_dict(federation_config.get("options"))
        ),
    )
    res = stub.StartRun(req)

    # Delete FAB file once it has been sent to the Exec API
    Path(fab_path).unlink()
    typer.secho(f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN)

    if stream:
        start_stream(res.run_id, channel, CONN_REFRESH_PERIOD)


def _run_without_exec_api(
    app: Optional[Path],
    federation_config: dict[str, Any],
    config_overrides: Optional[list[str]],
    federation: str,
) -> None:
    try:
        num_supernodes = federation_config["options"]["num-supernodes"]
        verbose: Optional[bool] = federation_config["options"].get("verbose")
        backend_cfg = federation_config["options"].get("backend", {})
    except KeyError as err:
        typer.secho(
            "❌ The project's `pyproject.toml` needs to declare the number of"
            " SuperNodes in the simulation. To simulate 10 SuperNodes,"
            " use the following notation:\n\n"
            f"[tool.flwr.federations.{federation}]\n"
            "options.num-supernodes = 10\n",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    command = [
        "flower-simulation",
        "--app",
        f"{app}",
        "--num-supernodes",
        f"{num_supernodes}",
    ]

    if backend_cfg:
        # Stringify as JSON
        command.extend(["--backend-config", json.dumps(backend_cfg)])

    if verbose:
        command.extend(["--verbose"])

    if config_overrides:
        command.extend(["--run-config", f"{' '.join(config_overrides)}"])

    # Run the simulation
    subprocess.run(
        command,
        check=True,
        text=True,
    )

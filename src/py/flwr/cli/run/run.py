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

import hashlib
import json
import subprocess
import sys
from logging import DEBUG
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from flwr.cli.build import build
from flwr.cli.config_utils import load_and_validate
from flwr.common.config import flatten_dict, parse_config_args
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


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
) -> None:
    """Run Flower App."""
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
            f"❌ There is no `{federation}` federation declared in "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if "address" in federation_config:
        _run_with_superexec(app, federation_config, config_overrides)
    else:
        _run_without_superexec(app, federation_config, config_overrides, federation)


def _run_with_superexec(
    app: Path,
    federation_config: dict[str, Any],
    config_overrides: Optional[list[str]],
) -> None:

    insecure_str = federation_config.get("insecure")
    if root_certificates := federation_config.get("root-certificates"):
        root_certificates_bytes = (app / root_certificates).read_bytes()
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
    stub = ExecStub(channel)

    fab_path = Path(build(app))
    content = fab_path.read_bytes()
    fab = Fab(hashlib.sha256(content).hexdigest(), content)

    req = StartRunRequest(
        fab=fab_to_proto(fab),
        override_config=user_config_to_proto(parse_config_args(config_overrides)),
        federation_config=user_config_to_proto(
            flatten_dict(federation_config.get("options"))
        ),
    )
    res = stub.StartRun(req)

    # Delete FAB file once it has been sent to the SuperExec
    fab_path.unlink()
    typer.secho(f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN)


def _run_without_superexec(
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

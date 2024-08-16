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

import subprocess
import sys
from logging import DEBUG
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_ssh_private_key,
    load_ssh_public_key,
)
from typing_extensions import Annotated

from flwr.cli.build import build
from flwr.cli.config_utils import load_and_validate
from flwr.cli.run.run_interceptor import RunInterceptor
from flwr.common.config import flatten_dict, parse_config_args
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.common.serde import user_config_to_proto
from flwr.proto.exec_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.exec_pb2_grpc import ExecStub


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
        Optional[List[str]],
        typer.Option(
            "--run-config",
            "-c",
            help="Override configuration key-value pairs, should be of the format:\n\n"
            "`--run-config key1=value1,key2=value2 --run-config key3=value3`\n\n"
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
    app: Optional[Path],
    federation_config: Dict[str, Any],
    config_overrides: Optional[List[str]],
) -> None:  # pylint: disable=R0914

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        log(DEBUG, channel_connectivity)

    insecure_str = federation_config.get("insecure")
    if root_certificates := federation_config.get("root-certificates"):
        root_certificates_bytes = Path(root_certificates).read_bytes()
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

    authentication_keys = _try_setup_user_authentication(
        federation_config.get("private_key_path"),
        federation_config.get("public_key_path"),
        federation_config.get("superexec_public_key_path"),
    )

    channel = create_channel(
        server_address=federation_config["address"],
        insecure=insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=(
            RunInterceptor(
                authentication_keys[0],
                authentication_keys[1],
                authentication_keys[2],
            )
            if authentication_keys is not None
            else None
        ),
    )
    channel.subscribe(on_channel_state_change)
    stub = ExecStub(channel)

    fab_path = Path(build(app))

    req = StartRunRequest(
        fab_file=fab_path.read_bytes(),
        override_config=user_config_to_proto(
            parse_config_args(config_overrides, separator=",")
        ),
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
    federation_config: Dict[str, Any],
    config_overrides: Optional[List[str]],
    federation: str,
) -> None:
    try:
        num_supernodes = federation_config["options"]["num-supernodes"]
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

    if config_overrides:
        command.extend(["--run-config", f"{','.join(config_overrides)}"])

    # Run the simulation
    subprocess.run(
        command,
        check=True,
        text=True,
    )


def _try_setup_user_authentication(
    private_key_path: Optional[str],
    public_key_path: Optional[str],
    superexec_public_key_path: Optional[str],
) -> Optional[
    Tuple[
        ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey, ec.EllipticCurvePublicKey
    ]
]:
    if not private_key_path and not public_key_path and not superexec_public_key_path:
        return None

    if not private_key_path or not public_key_path or not superexec_public_key_path:
        typer.secho(
            "❌ User authentication requires file paths to 'private_key_path', "
            "'public_key_path', and 'superexec_public_key_path' to be provided "
            "(providing only one of them is not sufficient).",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    try:
        ssh_private_key = load_ssh_private_key(
            Path(private_key_path).read_bytes(),
            None,
        )
        if not isinstance(ssh_private_key, ec.EllipticCurvePrivateKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm) as err:
        typer.secho(
            "❌ Error: Unable to parse the private key file in "
            "'private_key_path'. User authentication requires elliptic curve "
            "private and public key pair. Please ensure that the file "
            "path points to a valid private key file and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    try:
        ssh_public_key = load_ssh_public_key(Path(public_key_path).read_bytes())
        if not isinstance(ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm) as err:
        typer.secho(
            "❌ Error: Unable to parse the public key file in "
            "'public_key_path'. User authentication requires elliptic curve "
            "private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    try:
        superexec_ssh_public_key = load_ssh_public_key(
            Path(superexec_public_key_path).read_bytes()
        )
        if not isinstance(superexec_ssh_public_key, ec.EllipticCurvePublicKey):
            raise ValueError()
    except (ValueError, UnsupportedAlgorithm) as err:
        typer.secho(
            "❌ Error: Unable to parse the superexec public key file in "
            "'public_key'. User authentication requires elliptic curve "
            "private and public key pair. Please ensure that the file "
            "path points to a valid public key file and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1) from err

    return (ssh_private_key, ssh_public_key, superexec_ssh_public_key)
